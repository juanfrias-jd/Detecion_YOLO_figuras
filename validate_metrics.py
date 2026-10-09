"""
Calcula todas las métricas pedidas en el enunciado:
  - mAP@0.5, precisión y recall por clase (usando la validación nativa de Ultralytics)
  - Matriz de confusión
  - Accuracy_val personalizado: exactitud de clasificación sobre validación,
    considerando "correcta" una detección que empareja un Ground Truth con
    IoU >= 0.5 y clase correcta.

Guarda todo en models/metrics.json para que detect_realtime.py pueda mostrar
el Accuracy_val como un overlay FIJO durante la demo en vivo.

Uso:
    python validate_metrics.py
"""
import json
import os

import numpy as np
from ultralytics import YOLO

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/shapes_yolov8/weights/best.pt"
DATA_YAML = "data.yaml"
VAL_IMAGES_DIR = "data_merged/val/images"
VAL_LABELS_DIR = "data_merged/val/labels"
IOU_THRESHOLD = 0.5
OUTPUT_METRICS_PATH = "models/metrics.json"


# =============================
# Utilidades de IoU (igual a la fórmula del Taller 3)
# =============================
def yolo_to_xyxy(cls, xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return int(cls), [x1, y1, x2, y2]


def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


# =============================
# Accuracy_val personalizado
# =============================
def compute_accuracy_val(model, images_dir, labels_dir, iou_threshold=IOU_THRESHOLD):
    total_gt = 0
    correct = 0

    for img_name in sorted(os.listdir(images_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        result = model.predict(img_path, verbose=False, conf=0.25)[0]
        img_h, img_w = result.orig_shape

        # Ground truth de esta imagen
        gt_boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = (int(parts[0]), *map(float, parts[1:5]))
                cls_, box = yolo_to_xyxy(cls, xc, yc, w, h, img_w, img_h)
                gt_boxes.append((cls_, box))

        # Predicciones del modelo en esta imagen
        pred_boxes = []
        for b in result.boxes:
            cls_pred = int(b.cls.item())
            xyxy = b.xyxy[0].tolist()
            pred_boxes.append((cls_pred, xyxy))

        matched_pred_idx = set()
        for gt_cls, gt_box in gt_boxes:
            total_gt += 1
            best_iou, best_idx = 0.0, -1
            for idx, (p_cls, p_box) in enumerate(pred_boxes):
                if idx in matched_pred_idx or p_cls != gt_cls:
                    continue
                current_iou = iou(gt_box, p_box)
                if current_iou > best_iou:
                    best_iou, best_idx = current_iou, idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                correct += 1
                matched_pred_idx.add(best_idx)

    accuracy_val = correct / total_gt if total_gt > 0 else 0.0
    return accuracy_val, correct, total_gt


def main():
    print("📥 Cargando modelo...")
    model = YOLO(MODEL_PATH)

    # --- Métricas nativas de Ultralytics: mAP, precisión, recall, matriz de confusión ---
    print("\n🔎 Calculando mAP@0.5, precisión, recall y matriz de confusión...")
    val_results = model.val(data=DATA_YAML, split="val", iou=IOU_THRESHOLD, plots=True)

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    precision_per_class = val_results.box.p.tolist()
    recall_per_class = val_results.box.r.tolist()
    class_names = val_results.names

    print(f"\n✅ mAP@0.5      : {map50:.4f}")
    print(f"✅ mAP@0.5:0.95  : {map50_95:.4f}")
    for i, name in class_names.items():
        print(f"  Clase '{name}': precisión = {precision_per_class[i]:.4f} | recall = {recall_per_class[i]:.4f}")

    # La matriz de confusión y las curvas quedan guardadas automáticamente
    # como imágenes en models/shapes_yolov8/ (confusion_matrix.png, etc.)

    # --- Accuracy_val personalizado (definición del Taller: IoU>=0.5 y clase correcta) ---
    print(f"\n🔎 Calculando Accuracy_val (IoU >= {IOU_THRESHOLD} y clase correcta)...")
    accuracy_val, correct, total_gt = compute_accuracy_val(model, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    print(f"✅ Accuracy_val = {accuracy_val:.4f}  ({correct}/{total_gt} detecciones correctas)")

    # --- Guardar todo para usarlo como overlay fijo en detect_realtime.py ---
    metrics = {
        "map50": map50,
        "map50_95": map50_95,
        "precision_per_class": {class_names[i]: precision_per_class[i] for i in class_names},
        "recall_per_class": {class_names[i]: recall_per_class[i] for i in class_names},
        "accuracy_val": accuracy_val,
        "correct_detections": correct,
        "total_ground_truth": total_gt,
        "iou_threshold": IOU_THRESHOLD,
    }

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Métricas guardadas en {OUTPUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
