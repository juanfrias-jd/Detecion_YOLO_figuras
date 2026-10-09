# ShapeDetect-YOLOv8: Detección de Círculo, Cuadrado, Triángulo, Hexágono y Estrella en Tiempo Real

## Objetivo
Entrenar y desplegar un modelo YOLOv8 que detecte formas geométricas (círculo, cuadrado,
triángulo, hexágono, estrella) en tiempo real vía webcam, mostrando bounding box, etiqueta, confianza y el
Accuracy de validación (`Accuracy_val`) como overlay fijo.

## Estructura del proyecto
```
ShapeDetect_YOLOv8/
├── data/                       # Dataset REAL (fotos propias anotadas, formato YOLO)
├── synthetic_data/             # Dataset SINTÉTICO (generado por script)
├── data_merged/                # Dataset final: train/val/test combinados
├── models/                     # Pesos entrenados + métricas
├── generate_synthetic_shapes.py
├── merge_datasets.py
├── data.yaml
├── train.py
├── validate_metrics.py
├── detect_realtime.py
├── requirements.txt
└── README.md
```

## 1. Preparación de datos

### 1.1 Dataset real (`data/`)
Debes recolectar y anotar tus propias fotos de círculos, cuadrados, triángulos, hexágonos y estrellas
(físicos: fichas, tapas, dados, recortes de cartulina, etc.), en formato YOLO:
```
data/
├── images/
│   ├── foto001.jpg
│   └── ...
└── labels/
    ├── foto001.txt   # class x_center y_center width height (normalizado 0-1)
    └── ...
```
Recomendado: usa [Roboflow](https://roboflow.com) o `LabelImg` para anotar rápido y
exportar directamente en formato YOLOv8.

**Orden de clases (debe coincidir con `data.yaml`):**
`0 = circulo`, `1 = cuadrado`, `2 = triangulo`, `3 = hexagono`, `4 = estrella`

### 1.1.b Dataset real desde Roboflow Universe (alternativa a fotos propias)
Dataset: Aerovant, *object-detection* (Roboflow Universe, CC BY 4.0)
https://universe.roboflow.com/aerovant/object-detection-q5uab
```bash
# PowerShell: definir la API key (no se guarda en el código)
$env:ROBOFLOW_API_KEY = "tu_api_key"
python download_roboflow.py      # descarga en data_roboflow/
python import_roboflow.py        # conserva solo las 5 figuras y las deja en data/images y data/labels
```
`import_roboflow.py` reasigna las clases de Roboflow al orden del proyecto y descarta
las figuras que no usamos (cruz, casa, pentágono, etc.).

### 1.2 Dataset sintético
```bash
python generate_synthetic_shapes.py
```
Genera 400 imágenes por clase (2000 en total, 5 clases) con fondos variados, colores,
tamaños y rotaciones aleatorias.

### 1.3 Combinar y dividir (70/20/10)
```bash
python merge_datasets.py
```
Genera `data_merged/{train,val,test}/{images,labels}` combinando lo real y lo sintético.

## 2. Entrenamiento
```bash
# GPU NVIDIA (Windows): instalar PyTorch con CUDA ANTES de requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train.py
```
Usa `yolov8n.pt` como backbone preentrenado, `imgsz=640`, `epochs=40`, con
early stopping (`patience=10`). Los pesos quedan en `models/shapes_yolov8/weights/best.pt`.

## 3. Validación y métricas
```bash
python validate_metrics.py
```
Calcula e imprime:
- mAP@0.5 y mAP@0.5:0.95
- Precisión y recall por clase
- Matriz de confusión (guardada como imagen en `models/shapes_yolov8/`)
- **Accuracy_val**: exactitud considerando correcta una detección con IoU ≥ 0.5 y clase correcta

Todo se guarda en `models/metrics.json`, que usa `detect_realtime.py` para mostrar
el overlay fijo de Accuracy_val.

## 4. Demo en tiempo real
```bash
python detect_realtime.py
```
Abre la webcam y muestra: bounding box, etiqueta con figura + color, confianza, y el
`Accuracy_val` fijo en la esquina superior izquierda. Presiona `q` para salir.

## Reproducir todo desde cero
```bash
python download_roboflow.py
python import_roboflow.py
python generate_synthetic_shapes.py
python merge_datasets.py
python train.py
python validate_metrics.py
python detect_realtime.py
```
