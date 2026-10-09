"""
Toma el dataset descargado de Roboflow (carpeta data_roboflow, formato YOLOv8),
conserva SOLO las 5 clases del proyecto y las reasigna a nuestro orden:
    0 circulo, 1 cuadrado, 2 triangulo, 3 hexagono, 4 estrella
El resultado se guarda en data/images y data/labels (el "dataset real"),
listo para que merge_datasets.py lo mezcle con el sintético y haga el 70/20/10.

Uso:
    python import_roboflow.py
"""
import os
import shutil

import yaml

SRC_DIR = "data_roboflow"
OUT_IMAGES = os.path.join("data", "images")
OUT_LABELS = os.path.join("data", "labels")

# nombre (en minúsculas) en Roboflow -> id de nuestra clase
NAME_TO_ID = {
    "circulo": 0, "círculo": 0, "circle": 0,
    "cuadrado": 1, "square": 1,
    "triangulo": 2, "triángulo": 2, "triangle": 2,
    "hexagono": 3, "hexágono": 3, "hexagon": 3,
    "estrella": 4, "star": 4,
}
OUR_NAMES = ["circulo", "cuadrado", "triangulo", "hexagono", "estrella"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def to_bbox(values):
    """Devuelve (xc, yc, w, h). Acepta caja (4 valores) o polígono (pares x y)."""
    if len(values) == 4:
        return values
    xs, ys = values[0::2], values[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def main():
    with open(os.path.join(SRC_DIR, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    id_map = {}
    print("Clases en el dataset de Roboflow:")
    for i, n in enumerate(names):
        new = NAME_TO_ID.get(str(n).strip().lower())
        if new is not None:
            id_map[i] = new
        print(f"  {i:2d}  {n:15s} -> {OUR_NAMES[new] if new is not None else '(se descarta)'}")
    if not id_map:
        print("❌ Ninguna clase coincide con las 5 figuras. Revisa los nombres.")
        return

    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)
    kept = skipped = 0
    per_class = [0] * 5

    for split in ("train", "valid", "val", "test"):
        img_dir = os.path.join(SRC_DIR, split, "images")
        lbl_dir = os.path.join(SRC_DIR, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        p = line.split()
                        if len(p) < 5 or int(float(p[0])) not in id_map:
                            continue
                        new_id = id_map[int(float(p[0]))]
                        xc, yc, w, h = to_bbox([float(v) for v in p[1:]])
                        lines.append(f"{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        per_class[new_id] += 1
            if not lines:      # imagen sin ninguna de nuestras figuras: no sirve
                skipped += 1
                continue
            new_stem = f"rf_{split}_{stem}"
            shutil.copy(os.path.join(img_dir, img_name),
                        os.path.join(OUT_IMAGES, new_stem + os.path.splitext(img_name)[1]))
            with open(os.path.join(OUT_LABELS, new_stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            kept += 1

    print(f"\n✅ Imágenes importadas: {kept}   (descartadas por no tener nuestras figuras: {skipped})")
    print("Objetos por clase:")
    for n, c in zip(OUR_NAMES, per_class):
        print(f"  {n:10s}: {c}")
    print(f"\nGuardadas en '{OUT_IMAGES}' y '{OUT_LABELS}'. Siguiente: python merge_datasets.py")


if __name__ == "__main__":
    main()
