"""
Genera imágenes sintéticas de círculos, cuadrados, triángulos, hexágonos y estrellas con
anotaciones en formato YOLO (detección): class x_center y_center width height,
todo normalizado entre 0 y 1.

Uso:
    python generate_synthetic_shapes.py
"""
import os
import random

import cv2
import numpy as np

# =============================
# CONFIG
# =============================
IMG_SIZE = 640  # coincide con imgsz=640 sugerido en el enunciado
OUTPUT_DIR = "synthetic_data"
NUM_IMAGES_PER_CLASS = 400  # 400 x 5 clases = 2000 imágenes sintéticas

# IMPORTANTE: este orden de clases debe coincidir exactamente con data.yaml
CLASS_NAMES = ["circulo", "cuadrado", "triangulo", "hexagono", "estrella"]


# =============================
# Fondo aleatorio (igual que en el proyecto anterior)
# =============================
def random_background(img_size):
    choice = random.choice(["solid", "noise", "gradient"])
    if choice == "solid":
        color = np.random.randint(0, 255, 3).tolist()
        return np.full((img_size, img_size, 3), color, dtype=np.uint8)
    elif choice == "noise":
        return np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    else:  # gradient
        base = np.linspace(0, 255, img_size, dtype=np.uint8)
        grad = np.tile(base, (img_size, 1))
        return cv2.merge([grad, grad.T, grad])


def random_color():
    # Evitamos colores casi negros o casi blancos para que la forma resalte
    return tuple(int(c) for c in np.random.randint(30, 225, 3))


def rotate_points(pts, angle_deg, cx, cy):
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    rotated = pts @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


# =============================
# Dibujar cada forma y devolver su bounding box real (no solo un cuadrado fijo)
# =============================
def draw_shape(img, class_id, img_size):
    color = random_color()
    size = random.randint(70, 220)
    margin = size  # margen para que la forma no se salga del lienzo
    cx = random.randint(margin, img_size - margin)
    cy = random.randint(margin, img_size - margin)
    angle = random.uniform(0, 360)

    if class_id == 0:  # círculo (no rota, es simétrico)
        radius = size // 2
        cv2.circle(img, (cx, cy), radius, color, -1)
        x_min, y_min = cx - radius, cy - radius
        x_max, y_max = cx + radius, cy + radius

    elif class_id == 1:  # cuadrado (con rotación aleatoria)
        half = size / 2
        pts = np.array([[-half, -half], [half, -half],
                         [half, half], [-half, half]], dtype=np.float32)
        pts = rotate_points(pts, angle, cx, cy)
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    elif class_id == 2:  # triángulo equilátero (con rotación aleatoria)
        h = size * (np.sqrt(3) / 2)
        pts = np.array([[0, -2 / 3 * h],
                         [-size / 2, 1 / 3 * h],
                         [size / 2, 1 / 3 * h]], dtype=np.float32)
        pts = rotate_points(pts, angle, cx, cy)
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    elif class_id == 3:  # hexágono regular (6 vértices, con rotación aleatoria)
        r = size / 2
        ang = np.radians(np.arange(6) * 60)
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
        pts = rotate_points(pts, angle, cx, cy)
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    else:  # estrella de 5 puntas (10 vértices alternando radio externo/interno)
        r_out = size / 2
        r_in = r_out * 0.4
        ang = np.radians(np.arange(10) * 36 - 90)
        radii = np.where(np.arange(10) % 2 == 0, r_out, r_in)
        pts = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=1).astype(np.float32)
        pts = rotate_points(pts, angle, cx, cy)
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    # Recortar al lienzo (por si la rotación empuja algún vértice fuera)
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(img_size), x_max)
    y_max = min(float(img_size), y_max)

    x_center = (x_min + x_max) / 2 / img_size
    y_center = (y_min + y_max) / 2 / img_size
    w = (x_max - x_min) / img_size
    h_norm = (y_max - y_min) / img_size

    label = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h_norm:.6f}"
    return label


def create_shape_image(class_id, img_size=IMG_SIZE):
    img = random_background(img_size)
    label = draw_shape(img, class_id, img_size)
    return img, label


# =============================
# Generar el dataset completo
# =============================
def generate_dataset(output_dir, num_images_per_class):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    counter = 0
    for class_id in range(len(CLASS_NAMES)):
        for _ in range(num_images_per_class):
            img, label = create_shape_image(class_id)

            img_name = f"{counter:05d}.jpg"
            label_name = f"{counter:05d}.txt"
            cv2.imwrite(os.path.join(output_dir, "images", img_name), img)
            with open(os.path.join(output_dir, "labels", label_name), "w") as f:
                f.write(label + "\n")

            counter += 1

    print(f"✅ Dataset sintético generado en '{output_dir}'")
    print(f"Total de imágenes: {counter} ({num_images_per_class} por clase)")
    print(f"Clases: {CLASS_NAMES}")


if __name__ == "__main__":
    generate_dataset(OUTPUT_DIR, NUM_IMAGES_PER_CLASS)
