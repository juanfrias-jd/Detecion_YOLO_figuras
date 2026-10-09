"""
Combina el dataset REAL (carpeta data/) con el dataset SINTÉTICO
(carpeta synthetic_data/) generado por generate_synthetic_shapes.py,
y produce el dataset final dividido 70/20/10 en data_merged/{train,val,test}.

Uso:
    python merge_datasets.py
"""
import os
import random
import shutil

# =============================
# CONFIG
# =============================
REAL_DATA_DIR = "data"              # tus fotos reales anotadas (formato YOLO)
SYNTH_DATA_DIR = "synthetic_data"   # generado por generate_synthetic_shapes.py
OUTPUT_DIR = "data_merged"          # dataset final combinado

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =============================
# Buscar pares imagen + etiqueta en cualquier subcarpeta
# =============================
def find_images_and_labels(base_dir):
    items = []
    if not os.path.isdir(base_dir):
        print(f"⚠️  No se encontró la carpeta '{base_dir}', se omite.")
        return items

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                img_path = os.path.join(root, file)
                label_name = os.path.splitext(file)[0] + ".txt"
                # Soporta tanto estructura data/images+data/labels
                # como imágenes y labels en la misma carpeta
                label_path_same_dir = os.path.join(root, label_name)
                label_path_sibling = os.path.join(root.replace("images", "labels"), label_name)

                if os.path.exists(label_path_same_dir):
                    items.append((img_path, label_path_same_dir))
                elif os.path.exists(label_path_sibling):
                    items.append((img_path, label_path_sibling))
    return items


def copy_and_rename(src_img, src_label, dst_img_dir, dst_label_dir, idx):
    new_name = f"{idx:06d}"
    shutil.copy(src_img, os.path.join(dst_img_dir, new_name + ".jpg"))
    shutil.copy(src_label, os.path.join(dst_label_dir, new_name + ".txt"))


# =============================
# Fusionar y dividir
# =============================
def merge_datasets(real_dir, synth_dir, output_dir):
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

    real_imgs = find_images_and_labels(real_dir)
    synth_imgs = find_images_and_labels(synth_dir)

    print(f"📷 Imágenes reales encontradas:     {len(real_imgs)}")
    print(f"🖼️  Imágenes sintéticas encontradas: {len(synth_imgs)}")

    all_imgs = real_imgs + synth_imgs
    if len(all_imgs) == 0:
        print("❌ No se encontraron imágenes. Revisa la estructura de 'data/' y 'synthetic_data/'.")
        return

    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:],
    }

    idx = 0
    for split, items in splits.items():
        for img_path, label_path in items:
            dst_img_dir = os.path.join(output_dir, split, "images")
            dst_label_dir = os.path.join(output_dir, split, "labels")
            copy_and_rename(img_path, label_path, dst_img_dir, dst_label_dir, idx)
            idx += 1

    print("\n✅ Dataset combinado guardado en:", output_dir)
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
    print(f"  Total: {n_total}")


if __name__ == "__main__":
    merge_datasets(REAL_DATA_DIR, SYNTH_DATA_DIR, OUTPUT_DIR)
