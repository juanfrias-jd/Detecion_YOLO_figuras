"""
Detección en tiempo real con webcam: dibuja bounding box, etiqueta de clase,
tipo de figura, color dominante, confianza, y muestra el Accuracy_val (calculado previamente
por validate_metrics.py) como un overlay FIJO en la ventana de video.

Uso:
    python detect_realtime.py
"""
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/shapes_yolov8/weights/best.pt"
METRICS_PATH = "models/metrics.json"
CONF_THRESHOLD = 0.4
CAMERA_INDEX = 0  # cambia a 1 si usas una cámara externa/USB


def dominant_color_name(frame, box):
    """Nombre del color dominante dentro del bbox (zona central, en HSV)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    # Recorte central para evitar fondo en las esquinas del bbox
    roi = frame[max(0, y1 + h // 4):max(1, y2 - h // 4), max(0, x1 + w // 4):max(1, x2 - w // 4)]
    if roi.size == 0:
        return "?"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = [float(np.median(hsv[:, i])) for i in range(3)]
    if val < 50:
        return "negro"
    if sat < 40:
        return "blanco" if val > 180 else "gris"
    # Hue de OpenCV: 0-179
    if hue < 8 or hue >= 170:
        return "rojo"
    if hue < 20:
        return "naranja"
    if hue < 35:
        return "amarillo"
    if hue < 85:
        return "verde"
    if hue < 100:
        return "cian"
    if hue < 130:
        return "azul"
    if hue < 150:
        return "morado"
    return "rosa"


def draw_detections(frame, results, names):
    """Dibuja bbox + 'figura color confianza' sobre el frame."""
    out = frame.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        color_name = dominant_color_name(frame, (x1, y1, x2, y2))
        label = f"{names[cls]} {color_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def load_accuracy_val(metrics_path):
    if not os.path.exists(metrics_path):
        print("⚠️  No se encontró models/metrics.json — corre primero validate_metrics.py")
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics.get("accuracy_val")


def main():
    print("📥 Cargando modelo...")
    model = YOLO(MODEL_PATH)

    acc_val = load_accuracy_val(METRICS_PATH)
    acc_text = f"Acc_val={acc_val * 100:.0f}%" if acc_val is not None else "Acc_val=N/D"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    print("✅ Cámara iniciada. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar frame.")
            break

        # YOLOv8 recibe el frame en BGR directamente (así entrena por defecto Ultralytics,
        # no hace falta convertir a RGB manualmente en este caso).
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # bbox + etiqueta (figura + color + confianza)
        annotated_frame = draw_detections(frame, results, model.names)

        # Overlay FIJO del Accuracy_val, como pide el enunciado (p. ej. Acc_val=87%)
        cv2.rectangle(annotated_frame, (5, 5), (230, 45), (0, 0, 0), -1)
        cv2.putText(annotated_frame, acc_text, (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Deteccion de Formas Geometricas - YOLOv8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
