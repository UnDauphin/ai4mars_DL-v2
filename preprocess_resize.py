"""
preprocess_resize.py
====================
Prerredimensiona TODAS las imágenes y máscaras a 256×256 una sola vez.
Esto elimina el resize en cada epoch → el CPU solo lee y pasa datos.

Tiempo estimado: 10-20 minutos una sola vez.
Ahorro posterior: ~60-70% del tiempo de carga de datos por epoch.

Uso:
    python preprocess_resize.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST_IN  = Path("processed/manifest_clean.csv")
MANIFEST_OUT = Path("processed/manifest_256.csv")
IMAGES_DIR   = Path("data/images_256")
MASKS_DIR    = Path("data/masks_256")
IMG_SIZE     = 256

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Cargar manifest ───────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST_IN)
print(f"Total muestras: {len(df):,}")
print(f"Guardando imágenes en: {IMAGES_DIR}")
print(f"Guardando máscaras en: {MASKS_DIR}\n")

new_image_paths = []
new_mask_paths  = []
errors          = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resizing"):
    # ── Nombres de archivo de salida ──────────────────────────────────────────
    img_out  = IMAGES_DIR / f"{row['mission']}_{row['id']}.jpg"
    mask_out = MASKS_DIR  / f"{row['mission']}_{row['id']}.png"

    # Si ya existen, solo actualizar rutas (no reprocesar)
    if img_out.exists() and mask_out.exists():
        new_image_paths.append(str(img_out))
        new_mask_paths.append(str(mask_out))
        continue

    try:
        # Imagen → RGB → resize BILINEAR → JPEG
        img = Image.open(row["image_path"]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img.save(img_out, "JPEG", quality=95)

        # Máscara → L → resize NEAREST → PNG (preserva valores exactos)
        mask = Image.open(row["mask_path"]).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask.save(mask_out, "PNG")

        new_image_paths.append(str(img_out))
        new_mask_paths.append(str(mask_out))

    except Exception as e:
        errors.append((idx, str(e)))
        new_image_paths.append(row["image_path"])   # fallback a original
        new_mask_paths.append(row["mask_path"])

# ── Guardar nuevo manifest ────────────────────────────────────────────────────
df_out = df.copy()
df_out["image_path"] = new_image_paths
df_out["mask_path"]  = new_mask_paths
df_out.to_csv(MANIFEST_OUT, index=False)

print(f"\n✅ Manifest guardado en: {MANIFEST_OUT}")
print(f"   Imágenes procesadas: {len(df) - len(errors):,}")
print(f"   Errores:             {len(errors)}")
if errors:
    print("   Primeros errores:")
    for idx, e in errors[:5]:
        print(f"     idx={idx}: {e}")

# ── Verificar una muestra ─────────────────────────────────────────────────────
sample = df_out.sample(5, random_state=42)
print("\nVerificación de tamaños (muestra de 5):")
for _, row in sample.iterrows():
    img  = Image.open(row["image_path"])
    mask = Image.open(row["mask_path"])
    print(f"  img={img.size} mask={mask.size} mission={row['mission']}")

print("\n✅ Prerredimensionamiento completo.")
print(f"   Usa 'processed/manifest_256.csv' en todos los notebooks.")
