"""Demo en tiempo real del detector de emociones.

Lee la cámara, detecta la cara más grande, la reduce a 48 x 48 en gris
(como FER-2013) y la clasifica varias veces por segundo con:

- SVM sobre embeddings de CLIP (el mejor modelo del cuaderno),
- SVM sobre HOG (el mejor descriptor clásico),
- CLIP zero-shot con los prompts v3 del cuaderno.

Los SVM se entrenan con los mismos datos e hiperparámetros que develop.ipynb
y se guardan en results/ para no repetir el entrenamiento.

Uso:
    python demo_camara.py                 # primera cámara que funcione
    python demo_camara.py --camera 10     # /dev/video10
    python demo_camara.py --hz 3          # clasificaciones por segundo

Teclas: q/ESC salir · s suavizado on/off · f pantalla completa · p captura
"""

import argparse
import glob
import os
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import joblib
import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CLIP_CACHE = RESULTS_DIR / "clip_features.npz"
SVM_CLIP_PATH = RESULTS_DIR / "demo_svm_clip.joblib"
SVM_HOG_PATH = RESULTS_DIR / "demo_svm_hog.joblib"
YUNET_PATH = RESULTS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

SEED = 42
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLORS = {  # BGR
    "angry": (60, 60, 220),
    "disgust": (60, 160, 60),
    "fear": (170, 80, 150),
    "happy": (40, 200, 240),
    "neutral": (170, 170, 170),
    "sad": (200, 120, 40),
    "surprise": (40, 140, 250),
}

# Mismos prompts que la estrategia v3 del cuaderno.
PROMPTS_V1 = {
    "angry": ["a photo of a face looking angry"],
    "disgust": ["a photo of a face looking disgusted"],
    "fear": ["a photo of a face looking fearful"],
    "happy": ["a photo of a face looking happy"],
    "neutral": ["a photo of a face looking neutral"],
    "sad": ["a photo of a face looking sad"],
    "surprise": ["a photo of a face looking surprised"],
}
PROMPTS_V2 = {
    "angry": ["a close-up face of a furious, angry and mad person frowning"],
    "disgust": ["a close-up face of a disgusted, repulsed person grimacing in distaste"],
    "fear": ["a close-up face of a scared, terrified and anxious person in fear"],
    "happy": ["a close-up face of a joyful, cheerful person smiling happily"],
    "neutral": ["a close-up face of a calm, serious and expressionless neutral person"],
    "sad": ["a close-up face of an unhappy, gloomy and crying sad person"],
    "surprise": ["a close-up face of an astonished, shocked and amazed person with open mouth"],
}
PROMPTS_V3 = {
    c: PROMPTS_V1[c] + PROMPTS_V2[c] + [f"a grayscale portrait of a {c} facial expression"]
    for c in CLASSES
}


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def labels_from_paths(paths):
    return np.array([str(p).replace("\\", "/").split("/")[-2] for p in paths])


def find_image(path):
    """Busca una imagen del subconjunto en train/ o test/ del repositorio."""
    parts = str(path).replace("\\", "/").split("/")[-3:]
    for base in (ROOT, ROOT / "data"):
        candidate = base.joinpath(*parts)
        if candidate.is_file():
            return candidate
    return None


def make_svm(C, gamma):
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        ),
    )


def load_or_train_svm_clip():
    if SVM_CLIP_PATH.is_file():
        return joblib.load(SVM_CLIP_PATH)
    print("Entrenando SVM con CLIP (solo la primera vez)...", flush=True)
    cache = np.load(CLIP_CACHE, allow_pickle=True)
    X, y = cache["train"], labels_from_paths(cache["train_paths"])
    # Mejores hiperparámetros encontrados en el cuaderno: C=1, gamma=0.001.
    model = make_svm(C=1, gamma=0.001).fit(X, y)
    joblib.dump(model, SVM_CLIP_PATH)
    return model


def load_or_train_svm_hog():
    if SVM_HOG_PATH.is_file():
        return joblib.load(SVM_HOG_PATH)
    cache = np.load(CLIP_CACHE, allow_pickle=True)
    paths = [find_image(p) for p in cache["train_paths"]]
    if any(p is None for p in paths):
        print("No encuentro las imágenes de train: la demo seguirá sin HOG.")
        return None
    print("Entrenando SVM con HOG (solo la primera vez)...", flush=True)
    X = np.stack(
        [
            extract_hog(
                cv2.resize(
                    cv2.imread(str(p), cv2.IMREAD_GRAYSCALE),
                    (48, 48),
                    interpolation=cv2.INTER_AREA,
                )
            )
            for p in paths
        ]
    )
    y = labels_from_paths(cache["train_paths"])
    # Mejores hiperparámetros encontrados en el cuaderno: C=10, gamma="scale".
    model = make_svm(C=10, gamma="scale").fit(X, y)
    joblib.dump(model, SVM_HOG_PATH)
    return model


class EmotionModels:
    def __init__(self):
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        with warnings.catch_warnings():
            # Mismo modelo que en el cuaderno, para que los embeddings coincidan
            # con los que usamos para entrenar el SVM.
            warnings.simplefilter("ignore")
            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            )
        self.clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        with torch.inference_mode():
            text = []
            for c in CLASSES:
                emb = self.clip_model.encode_text(tokenizer(PROMPTS_V3[c])).float()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                text.append((mean / mean.norm()).numpy())
        self.text_embeddings = np.stack(text)

        self.svm_clip = load_or_train_svm_clip()
        self.svm_hog = load_or_train_svm_hog()

    def _proba(self, model, features):
        proba = model.predict_proba(features)[0]
        order = [list(model.classes_).index(c) for c in CLASSES]
        return proba[order]

    def predict(self, face_48):
        with torch.inference_mode():
            batch = self.clip_preprocess(Image.fromarray(face_48).convert("RGB"))
            emb = self.clip_model.encode_image(batch.unsqueeze(0)).float()
            emb = (emb / emb.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)

        logits = 100.0 * (emb @ self.text_embeddings.T)[0]
        zs = np.exp(logits - logits.max())
        zs /= zs.sum()

        result = {
            "svm_clip": self._proba(self.svm_clip, emb),
            "zero_shot": zs,
            "svm_hog": None,
        }
        if self.svm_hog is not None:
            result["svm_hog"] = self._proba(
                self.svm_hog, extract_hog(face_48).reshape(1, -1)
            )
        return result


# ----------------------------------------------------------------------------
# Cámara y cara
# ----------------------------------------------------------------------------
def open_camera(index):
    candidates = [index] if index is not None else sorted(
        int(p.replace("/dev/video", "")) for p in glob.glob("/dev/video*")
    ) or [0]
    for i in candidates:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2) if os.name != "nt" else cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            print(f"Cámara abierta: índice {i}")
            return cap
        cap.release()
    raise SystemExit(f"No se pudo abrir ninguna cámara ({candidates}). Prueba con --camera N.")


def yunet_model():
    """Detector de caras YuNet de OpenCV (OpenCV 5 ya no trae las cascadas Haar)."""
    if not YUNET_PATH.is_file():
        print("Descargando el detector de caras YuNet...", flush=True)
        response = requests.get(YUNET_URL, timeout=(15, 60))
        response.raise_for_status()
        YUNET_PATH.write_bytes(response.content)
    return str(YUNET_PATH)


class FaceFinder:
    def __init__(self, margin=1.25):
        self.detector = cv2.FaceDetectorYN.create(
            yunet_model(), "", (320, 320), score_threshold=0.7
        )
        self.margin = margin
        self.last_box = None

    def find(self, frame):
        scale = min(1.0, 640 / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4] / scale

        # Recorte cuadrado algo más amplio que la detección, parecido a FER-2013.
        side = max(w, h) * self.margin
        cx, cy = x + w / 2, y + h / 2 + 0.03 * h
        box = np.array([cx - side / 2, cy - side / 2, side, side])
        if self.last_box is not None:  # evita que el recuadro tiemble
            box = 0.6 * box + 0.4 * self.last_box
        self.last_box = box
        return box.astype(int)


def crop_face(gray, box):
    x, y, s, _ = box
    H, W = gray.shape
    pad = max(0, -x, -y, x + s - W, y + s - H)
    if pad:
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x, y = x + pad, y + pad
    crop = gray[y : y + s, x : x + s]
    return cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Clasificación en segundo plano
# ----------------------------------------------------------------------------
class Classifier(threading.Thread):
    def __init__(self, models, hz):
        super().__init__(daemon=True)
        self.models = models
        self.period = 1.0 / hz
        self.lock = threading.Lock()
        self.face = None
        self.result = None
        self.smoothed = None
        self.smoothing = True
        self.latency_ms = 0.0
        self.running = True

    def submit(self, face):
        with self.lock:
            self.face = face

    def clear(self):
        with self.lock:
            self.face = None
            self.result = None
            self.smoothed = None

    def snapshot(self):
        with self.lock:
            return self.smoothed if self.smoothing else self.result

    def run(self):
        while self.running:
            start = time.perf_counter()
            with self.lock:
                face = self.face
            if face is not None:
                result = self.models.predict(face)
                with self.lock:
                    if self.face is not None:  # la cara no se ha perdido mientras
                        self.result = result
                        if self.smoothed is None:
                            self.smoothed = result
                        else:
                            self.smoothed = {
                                k: None if v is None else 0.5 * v + 0.5 * self.smoothed[k]
                                for k, v in result.items()
                            }
                self.latency_ms = 1000 * (time.perf_counter() - start)
            time.sleep(max(0.0, self.period - (time.perf_counter() - start)))


# ----------------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.6, color=(240, 240, 240), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_bars(panel, title, proba, y0, width):
    put(panel, title, (20, y0), 0.65, (255, 255, 255), 2)
    best = int(np.argmax(proba))
    for i, c in enumerate(CLASSES):
        y = y0 + 16 + i * 26
        bar = int((width - 150) * proba[i])
        cv2.rectangle(panel, (100, y), (100 + bar, y + 18), COLORS[c], -1)
        cv2.rectangle(panel, (100, y), (width - 50, y + 18), (90, 90, 90), 1)
        weight = 2 if i == best else 1
        put(panel, c, (20, y + 14), 0.5, (255, 255, 255), weight)
        put(panel, f"{proba[i]:.0%}", (width - 45, y + 14), 0.45)
    return y0 + 16 + len(CLASSES) * 26 + 30


def render(frame, box, face, result, clf, width=440):
    view = cv2.flip(frame, 1)  # efecto espejo solo para mostrar
    H, W = view.shape[:2]
    panel = np.full((H, width, 3), 28, np.uint8)

    if box is not None:
        x, y, s, _ = box
        x = W - x - s  # coordenadas del recuadro en la imagen espejada
        label = CLASSES[int(np.argmax(result["svm_clip"]))] if result else "..."
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(view, (x, y), (x + s, y + s), color, 3)
        put(view, label.upper(), (x, max(30, y - 12)), 1.1, color, 3)
    else:
        put(view, "No se detecta ninguna cara", (30, 50), 1.0, (80, 80, 255), 2)

    put(panel, "Lo que ve el modelo (48x48)", (20, 30), 0.55)
    if face is not None:
        big = cv2.resize(face, (144, 144), interpolation=cv2.INTER_NEAREST)
        panel[45:189, 20:164] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    else:
        cv2.rectangle(panel, (20, 45), (164, 189), (90, 90, 90), 1)

    y = 230
    if result is not None:
        y = draw_bars(panel, "SVM + CLIP", result["svm_clip"], y, width)
        y = draw_bars(panel, "CLIP zero-shot (v3)", result["zero_shot"], y, width)
        if result["svm_hog"] is not None:
            hog_label = CLASSES[int(np.argmax(result["svm_hog"]))]
            put(panel, "SVM + HOG:", (20, y), 0.65, (255, 255, 255), 2)
            put(panel, hog_label, (160, y), 0.75, COLORS[hog_label], 2)
            y += 30

    status = (
        f"{1 / clf.period:.1f} clasif/s | {clf.latency_ms:.0f} ms | "
        f"suavizado {'on' if clf.smoothing else 'off'}"
    )
    put(panel, status, (20, H - 40), 0.45, (180, 180, 180))
    put(panel, "q salir  s suavizado  f pantalla  p captura", (20, H - 15), 0.45, (180, 180, 180))
    return np.hstack([view, panel])


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=None, help="índice de la cámara")
    parser.add_argument("--hz", type=float, default=2.5, help="clasificaciones por segundo")
    args = parser.parse_args()

    print("Cargando modelos...", flush=True)
    models = EmotionModels()
    cap = open_camera(args.camera)
    finder = FaceFinder()
    clf = Classifier(models, args.hz)
    clf.start()

    window = "Detector de emociones"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    fullscreen = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("La cámara ha dejado de enviar imágenes.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = finder.find(frame)
            face = None
            if box is None:
                finder.last_box = None
                clf.clear()
            else:
                face = crop_face(gray, box)
                clf.submit(face)

            canvas = render(frame, box, face, clf.snapshot(), clf)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                clf.smoothing = not clf.smoothing
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            if key == ord("p"):
                name = ROOT / f"captura_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(name), canvas)
                print("Captura guardada en", name)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        clf.running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
