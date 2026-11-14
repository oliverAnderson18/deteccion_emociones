import sys
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from skimage.feature import hog

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def pasar_a_48x48(img_gray):
    h, w = img_gray.shape
    lado = min(h, w)
    y0 = (h - lado) // 2
    x0 = (w - lado) // 2
    recorte = img_gray[y0 : y0 + lado, x0 : x0 + lado]
    return cv2.resize(recorte, (48, 48), interpolation=cv2.INTER_AREA)


def main():
    if len(sys.argv) < 2:
        print("Uso: python clasificar_foto.py <ruta_a_tu_foto.jpg>")
        sys.exit(1)

    foto_path = Path(sys.argv[1])
    img_bgr = cv2.imread(str(foto_path))
    if img_bgr is None:
        print(f"No se pudo abrir la imagen: {foto_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cara_48 = pasar_a_48x48(img_gray)

    # 1. Predicción con HOG + SVM
    feat_hog = hog(
        cara_48,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).astype(np.float32).reshape(1, -1)

    svm_hog = joblib.load("features/svm_hog.pkl")
    pred_hog = CLASSES[svm_hog.predict(feat_hog)[0]]

    # 2. Sacamos el embedding de CLIP de la cara 48*48
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_model.eval()

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    with torch.no_grad():
        t = torch.from_numpy(cara_48).unsqueeze(0).unsqueeze(0).float() / 255.0
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
        t = (t.repeat(1, 3, 1, 1) - mean) / std
        emb_clip = clip_model.encode_image(t)
        emb_clip = (emb_clip / emb_clip.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)

    # Predicción con CLIP + SVM
    svm_clip = joblib.load("features/svm_clip.pkl")
    pred_clip_svm = CLASSES[svm_clip.predict(emb_clip)[0]]

    # 3. Predicción con CLIP Zero-Shot
    prompts = [f"a photo of a face looking {c}" for c in CLASSES]
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokenizer(prompts))
        text_emb = (text_emb / text_emb.norm(dim=-1, keepdim=True)).numpy()
    probs_zs = 100.0 * (emb_clip @ text_emb.T)[0]
    probs_zs = np.exp(probs_zs - probs_zs.max())
    probs_zs /= probs_zs.sum()
    pred_clip_zs = CLASSES[probs_zs.argmax()]

    print(f"Foto: {foto_path}")
    print(f"  -> SVM (HOG)       : {pred_hog}")
    print(f"  -> SVM (CLIP)      : {pred_clip_svm}")
    print(f"  -> CLIP Zero-Shot  : {pred_clip_zs} ({probs_zs.max():.0%})")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Foto original")
    axes[0].axis("off")

    axes[1].imshow(cara_48, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Cara 48*48\nSVM (CLIP): {pred_clip_svm}\nSVM (HOG): {pred_hog}\nCLIP Zero-Shot: {pred_clip_zs}",
        fontsize=9,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("prediccion_foto.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
