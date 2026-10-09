"""
Entrena un modelo YOLOv8 para detectar círculo, cuadrado, triángulo, hexágono y estrella.

Uso:
    python train.py
"""
import torch
from ultralytics import YOLO

# =============================
# CONFIG
# =============================
BASE_MODEL = "yolov8n.pt"   # backbone preentrenado (nano: rápido, ideal para tiempo real)
DATA_YAML = "data.yaml"
IMG_SIZE = 640
EPOCHS = 40                 # dentro del rango sugerido 30-50
BATCH = 16                  # ajustar según la GPU disponible (bajar si hay error de memoria)
PATIENCE = 10                # early stopping: se detiene si no mejora en 10 épocas
PROJECT_DIR = "models"
RUN_NAME = "shapes_yolov8"


def main():
    # Usa la GPU (CUDA) si está disponible; si no, avisa y usa CPU
    if torch.cuda.is_available():
        device = 0
        print(f"🚀 Entrenando en GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  CUDA no disponible: se entrenará en CPU (lento). Revisa la instalación de PyTorch con GPU.")

    # Cargamos un modelo YOLOv8 preentrenado en COCO como backbone/punto de partida
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=device,
        exist_ok=True,
        plots=True,          # genera automáticamente curvas de pérdida/mAP
    )

    print("\n✅ Entrenamiento finalizado.")
    print(f"Pesos guardados en: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
