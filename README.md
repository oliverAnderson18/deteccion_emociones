# Proyecto 8 - Detector de emociones

**Grupo M:** Makar Isaev y Oliver Anderson Llorens

Clasificación de expresiones faciales en imágenes de 48 × 48 píxeles. Comparamos los errores de los modelos con nuestras anotaciones de 60 caras.

[Abrir `develop.ipynb` en Google Colab](https://colab.research.google.com/github/oliverAnderson18/deteccion_emociones/blob/master/develop.ipynb)

## Datos

**Licencia de los datos (según Kaggle):** [Open Database License 1.0 (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/) para la base de datos y [Database Contents License 1.0 (DbCL)](https://opendatacommons.org/licenses/dbcl/1-0/) para su contenido.

[FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013) contiene 35.887 imágenes en gris de siete emociones. [Descargar el ZIP de OneDrive](https://upm365-my.sharepoint.com/:u:/g/personal/m_isaev_alumnos_upm_es/IQCCEo6WLS8MQo8v_WJq3puyAXBRnYVKuuZGVitAjMBJjH0?e=KRgXA5&download=1).

Eliminamos hashes nulos y repetidos con dHash. Si un hash aparece en ambos conjuntos, conservamos la imagen de test. El subconjunto final tiene 6.456 imágenes (5.076 de train y 1.380 de test): 1.000 por emoción, salvo `disgust`, con las 456 que quedan tras la limpieza.

## Método y resultados

Comparamos HOG, LBP y ORB con embeddings de ResNet50 y CLIP ViT-B/32. Para clasificar usamos SVM con validación cruzada en train y kNN con FAISS (`k=15`). El cuaderno incluye matrices de confusión, t-SNE y tiempos frente a búsqueda exhaustiva.

| Descriptor | SVM, exactitud equilibrada | kNN con FAISS exacto, exactitud equilibrada |
|---|---:|---:|
| CLIP | 63,8 % | 56,0 % |
| ResNet50 | 45,2 % | 35,3 % |
| HOG | 43,9 % | 40,0 % |
| LBP | 31,9 % | 25,7 % |
| ORB | 25,3 % | 20,9 % |

En CLIP zero-shot, la exactitud equilibrada fue del 33,9 % con adjetivos, del 37,5 % con sinónimos y gestos y del 41,2 % al combinar los prompts.

El acuerdo entre anotadores fue κ = 0,901 (κ = 0,572 y 0,553 frente a FER-2013). Humanos y modelos comparten confusiones como `angry ↔ disgust`, `fear ↔ surprise` y `neutral ↔ sad`. En 19 de las 60 caras, los dos anotadores eligieron la misma emoción y esta difería de la etiqueta oficial.

## Cómo reproducir el trabajo

1. Abre el cuaderno en Colab y ejecuta esta celda al inicio:

   ```python
   !git clone https://github.com/oliverAnderson18/deteccion_emociones.git
   %cd deteccion_emociones
   %pip install -q -r requirements.txt
   ```

2. Ejecuta el cuaderno de principio a fin. Si faltan las carpetas `train` y `test`, se descargan desde OneDrive.

En local, instala las dependencias con `python -m pip install -r requirements.txt` y abre el cuaderno desde la raíz del repositorio. Los CSV de anotaciones usan los índices del test original. Si cambias el subconjunto, usa una carpeta de resultados nueva para recalcular los embeddings.
