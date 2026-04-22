"""
mars_utils.py
=============
Infraestructura compartida para los 5 notebooks de benchmark.
Todos los modelos importan desde aquí para garantizar:
  - Mismo split (split_indices.pkl)
  - Mismo preprocessing
  - Mismas métricas
  - Mismo training loop
  - Mismo formato de guardado de resultados
"""

import os
import random
import pickle
import time
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 0. CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES   = 4
IGNORE_INDEX  = 255
SEED          = 42
IMG_SIZE      = 256
BATCH_SIZE    = 16

CLASS_NAMES   = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}

# Normalización ImageNet (canal gris replicado ×3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PROCESSED_DIR   = Path("processed")
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR     = Path("results")

for d in [PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SPLIT_INDICES_PATH   = PROCESSED_DIR / "split_indices.pkl"
BENCHMARK_CSV_PATH   = RESULTS_DIR   / "benchmark_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEED CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMS — mismas en todos los notebooks
# ─────────────────────────────────────────────────────────────────────────────

class JointTransformTrain:
    """
    Aplica la misma transformación geométrica a imagen Y máscara.
    Augmentaciones solo para train (nunca para val/test).
    """
    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size  = img_size
        self.color_jit = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Resize (imagen: BILINEAR, máscara: NEAREST)
        img  = TF.resize(img,  [self.img_size, self.img_size], T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], T.InterpolationMode.NEAREST)

        # Flip horizontal
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Flip vertical
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # Rotación aleatoria ±15°
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=T.InterpolationMode.BILINEAR,
                          fill=0)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST,
                          fill=IGNORE_INDEX)

        # Color jitter (solo imagen)
        img = self.color_jit(img)

        # ToTensor + Normalize
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


class JointTransformVal:
    """Solo resize + normalize. Sin augmentaciones."""
    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size  = img_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image, mask: Image.Image):
        img  = TF.resize(img,  [self.img_size, self.img_size], T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], T.InterpolationMode.NEAREST)
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MarsTerrainDataset(Dataset):
    """
    Dataset para AI4MARS.
    - Imagen: PIL → RGB (canal gris replicado ×3 para backbones ImageNet).
    - Máscara: PIL → L, valores {0,1,2,3,255}. NO se remap el 255.
    - ignore_index=255 se maneja en la loss y métricas.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask, row["mission"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT ESTRATIFICADO — guardado/cargado desde pkl
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = SEED,
    save_path: Path = SPLIT_INDICES_PATH,
) -> tuple:
    """
    Si split_indices.pkl existe → carga y retorna los mismos splits.
    Si no existe → genera, guarda y retorna.
    Garantiza que TODOS los notebooks usen exactamente el mismo split.
    """
    if save_path.exists():
        with open(save_path, "rb") as f:
            indices = pickle.load(f)
        df_train = df.iloc[indices["train"]].reset_index(drop=True)
        df_val   = df.iloc[indices["val"]  ].reset_index(drop=True)
        df_test  = df.iloc[indices["test"] ].reset_index(drop=True)
        print(f"✅ Split cargado desde {save_path}")
    else:
        train_f, val_f, test_f = [], [], []
        rng = np.random.default_rng(seed)

        for _, group in df.groupby("mission"):
            g = group.sample(frac=1, random_state=seed)
            n = len(g)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)
            train_f.append(g.iloc[:n_tr])
            val_f.append(  g.iloc[n_tr : n_tr + n_va])
            test_f.append( g.iloc[n_tr + n_va :])

        df_train = pd.concat(train_f).sample(frac=1, random_state=seed).reset_index(drop=True)
        df_val   = pd.concat(val_f  ).sample(frac=1, random_state=seed).reset_index(drop=True)
        df_test  = pd.concat(test_f ).sample(frac=1, random_state=seed).reset_index(drop=True)

        # Verificar no data leakage (ids únicos)
        train_ids = set(df_train["id"].values)
        test_ids  = set(df_test["id"].values)
        assert len(train_ids & test_ids) == 0, "¡Data leakage! IDs compartidos entre train y test."

        indices = {
            "train": df_train.index.tolist(),
            "val":   df_val.index.tolist(),
            "test":  df_test.index.tolist(),
        }
        with open(save_path, "wb") as f:
            pickle.dump(indices, f)
        print(f"✅ Split generado y guardado en {save_path}")

    print(f"Train: {len(df_train):>6}  {df_train['mission'].value_counts().to_dict()}")
    print(f"Val  : {len(df_val):>6}  {df_val['mission'].value_counts().to_dict()}")
    print(f"Test : {len(df_test):>6}  {df_test['mission'].value_counts().to_dict()}")
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. WEIGHTED SAMPLER por misión
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts  = df["mission"].value_counts().to_dict()
    weights = torch.DoubleTensor(
        df["mission"].map(lambda m: 1.0 / counts[m]).values
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    df_train, df_val, df_test,
    img_size:    int  = IMG_SIZE,
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = 0,
    balance_missions: bool = True,
):
    train_ds = MarsTerrainDataset(df_train, JointTransformTrain(img_size))
    val_ds   = MarsTerrainDataset(df_val,   JointTransformVal(img_size))
    test_ds  = MarsTerrainDataset(df_test,  JointTransformVal(img_size))

    sampler = make_weighted_sampler(df_train) if balance_missions else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 7. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:   torch.Tensor,   # [B, C, H, W] logits  o  [B, H, W] clases
    targets: torch.Tensor,   # [B, H, W] LongTensor
    num_classes:  int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """
    Retorna: mIoU, IoU_per_class, pixel_accuracy.
    Excluye píxeles con valor ignore_index.
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    valid   = targets != ignore_index
    preds   = preds[valid]
    targets = targets[valid]

    correct = int((preds == targets).sum())
    total   = int(valid.sum())

    iou_per_class = {}
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        denom = tp + fp + fn
        iou_per_class[CLASS_NAMES[c]] = tp / denom if denom > 0 else 0.0

    return {
        "mIoU":          float(np.mean(list(iou_per_class.values()))),
        "IoU_per_class": iou_per_class,
        "pixel_accuracy": correct / max(total, 1),
    }


class MetricsAccumulator:
    """Acumula TP/FP/FN a lo largo de un epoch entero."""
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.correct = 0
        self.total   = 0

    def update(self, preds, targets):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        valid = t != self.ignore_index
        p, t  = p[valid], t[valid]
        self.correct += int((p == t).sum())
        self.total   += int(valid.sum())
        for c in range(self.num_classes):
            self.tp[c] += int(((p == c) & (t == c)).sum())
            self.fp[c] += int(((p == c) & (t != c)).sum())
            self.fn[c] += int(((p != c) & (t == c)).sum())

    def compute(self):
        iou = {}
        for c in range(self.num_classes):
            d = self.tp[c] + self.fp[c] + self.fn[c]
            iou[CLASS_NAMES[c]] = float(self.tp[c] / d) if d > 0 else 0.0
        return {
            "mIoU":           float(np.mean(list(iou.values()))),
            "IoU_per_class":  iou,
            "pixel_accuracy": self.correct / max(self.total, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        valid   = targets != self.ignore_index
        dice    = 0.0
        n_pres  = 0
        for c in range(self.num_classes):
            t = (targets[valid] == c).float()
            if t.sum() == 0:
                continue
            p = probs[:, c][valid]
            inter = (p * t).sum()
            dice += (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            n_pres += 1
        return 1.0 - dice / max(n_pres, 1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0,
                 num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)           # [B, H, W]
        pt      = torch.exp(-ce_loss)
        focal   = self.alpha * (1 - pt) ** self.gamma * ce_loss
        mask    = targets != self.ignore_index
        return focal[mask].mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice — usado por MarsSeg."""
    def __init__(self, alpha=0.25, gamma=2.0,
                 num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, num_classes, ignore_index)
        self.dice  = DiceLoss(num_classes, ignore_index)

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINING LOOP GENÉRICO
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, aux_weight=0.0):
    model.train()
    total_loss = 0.0
    acc = MetricsAccumulator()
    scaler = torch.cuda.amp.GradScaler(enabled=True)   # ← AMP

    pbar = tqdm(loader, desc="  train", leave=False)
    for imgs, masks, _ in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():                     # ← AMP
            output = model(imgs)
            if isinstance(output, dict):
                logits = output["out"]
                loss = criterion(logits, masks)
                if aux_weight > 0 and "aux" in output:
                    loss = loss + aux_weight * criterion(output["aux"], masks)
            else:
                logits = output
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        acc.update(logits.detach(), masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    results = acc.compute()
    results["loss"] = total_loss / len(loader)
    return results


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    acc = MetricsAccumulator()

    with torch.cuda.amp.autocast():                         # ← AMP
        for imgs, masks, _ in tqdm(loader, desc="   val ", leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            output = model(imgs)
            logits = output["out"] if isinstance(output, dict) else output
            total_loss += criterion(logits, masks).item()
            acc.update(logits, masks)

    results = acc.compute()
    results["loss"] = total_loss / len(loader)
    return results


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    model_name:  str = "model",
    num_epochs:  int = 30,
    patience:    int = 10,
    aux_weight: float = 0.0,
    seeds: list  = [42, 123, 7],
):
    """
    Entrena con N seeds (mín. 3, requisito del entregable).
    Retorna lista de resultados por seed + resumen estadístico.
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    all_test_results = []

    # Necesitamos el test loader — se pasa como atributo del train_loader
    # En la práctica se llama directamente desde el notebook

    best_state   = None
    best_miou    = -1.0
    no_improve   = 0
    history      = {"train": [], "val": []}
    t_start      = time.time()

    print(f"\n{'='*55}")
    print(f"Entrenando: {model_name}  |  device={device}")
    print(f"{'='*55}")

    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"{model_name}",
                     unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                             device, aux_weight=aux_weight)
        va = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["mIoU"])
            else:
                scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)

        # Actualiza la barra con las métricas clave
        epoch_bar.set_postfix(
            loss=f"{tr['loss']:.4f}",
            mIoU=f"{tr['mIoU']:.4f}",
            val_mIoU=f"{va['mIoU']:.4f}",
            big_rock=f"{va['IoU_per_class']['big_rock']:.4f}",
            best=f"{best_miou:.4f}",
        )

        if va["mIoU"] > best_miou:
            best_miou  = va["mIoU"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": best_state,
                        "val_metrics": va},
                       CHECKPOINTS_DIR / f"{model_name}_best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"  ⏹ Early stopping en epoch {epoch} (best val mIoU={best_miou:.4f})")
                break

    train_time = time.time() - t_start
    tqdm.write(f"  ✅ Mejor val mIoU = {best_miou:.4f}  (epoch {best_epoch})  |  Tiempo: {train_time:.0f}s")

    # Cargar mejor estado
    model.load_state_dict(best_state)
    # Guardar state_dict final
    torch.save(model.state_dict(),
               CHECKPOINTS_DIR / f"{model_name}_final_state_dict.pth")

    return history, best_miou, best_epoch, train_time


# ─────────────────────────────────────────────────────────────────────────────
# 10. MULTI-RUN (mínimo 3 seeds — requisito del entregable)
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(
    model_fn,          # callable() → modelo nuevo
    df_train, df_val, df_test,
    criterion_fn,      # callable() → criterio nuevo
    optimizer_fn,      # callable(params) → optimizer
    scheduler_fn,      # callable(optimizer) → scheduler
    model_name: str,
    device: str,
    seeds: list      = [42, 123, 7],
    num_epochs: int  = 30,
    patience: int    = 10,
    batch_size: int  = BATCH_SIZE,
    img_size: int    = IMG_SIZE,
    num_workers: int = 0,
    aux_weight: float = 0.0,
):
    """
    Ejecuta N seeds y retorna media ± std de métricas de TEST.
    """
    all_test = []

    for seed in seeds:
        set_seed(seed)
        tqdm.write(f"\n{'─'*50}")
        tqdm.write(f"  Seed {seed}  |  {model_name}")

        train_loader, val_loader, test_loader = build_dataloaders(
            df_train, df_val, df_test,
            img_size=img_size, batch_size=batch_size,
            num_workers=num_workers, balance_missions=True,
        )

        model     = model_fn().to(device)
        criterion = criterion_fn()
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        name_seed = f"{model_name}_seed{seed}"
        history, best_miou, best_epoch, train_time = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion, device,
            model_name=name_seed,
            num_epochs=num_epochs, patience=patience,
            aux_weight=aux_weight,
        )

        # Evaluación en test
        test_res = evaluate(model, test_loader, criterion, device)
        test_res["seed"]       = seed
        test_res["best_epoch"] = best_epoch
        test_res["train_time"] = train_time
        all_test.append(test_res)
        tqdm.write(f"  Test mIoU = {test_res['mIoU']:.4f}")

    # Resumen estadístico
    miou_vals = [r["mIoU"] for r in all_test]
    summary = {
        "model":          model_name,
        "mIoU_mean":      float(np.mean(miou_vals)),
        "mIoU_std":       float(np.std(miou_vals)),
        "mIoU_ci95":      float(1.96 * np.std(miou_vals) / np.sqrt(len(miou_vals))),
        "pixel_acc_mean": float(np.mean([r["pixel_accuracy"] for r in all_test])),
        "train_time_mean": float(np.mean([r["train_time"] for r in all_test])),
        "best_epoch_mean": float(np.mean([r["best_epoch"] for r in all_test])),
        "per_seed":       all_test,
    }
    for c_name in CLASS_NAMES.values():
        vals = [r["IoU_per_class"][c_name] for r in all_test]
        summary[f"iou_{c_name}_mean"] = float(np.mean(vals))
        summary[f"iou_{c_name}_std"]  = float(np.std(vals))

    print(f"\n{'='*55}")
    print(f"RESUMEN {model_name}")
    print(f"  mIoU : {summary['mIoU_mean']:.4f} ± {summary['mIoU_std']:.4f}"
          f"  IC95: ±{summary['mIoU_ci95']:.4f}")
    for c in CLASS_NAMES.values():
        print(f"  IoU({c}): "
              f"{summary[f'iou_{c}_mean']:.4f} ± {summary[f'iou_{c}_std']:.4f}")
    print(f"{'='*55}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 11. GUARDADO EN benchmark_results.csv (append)
# ─────────────────────────────────────────────────────────────────────────────

def append_benchmark_results(
    model_name:   str,
    best_epoch:   float,
    val_miou:     float,
    test_miou:    float,
    test_miou_std: float,
    test_miou_ci95: float,
    test_acc:     float,
    iou_soil:     float,
    iou_bedrock:  float,
    iou_sand:     float,
    iou_bigrock:  float,
    params_M:     float,
    train_time_s: float,
):
    """Append de resultados al CSV compartido. Crea el header si no existe."""
    csv_path = BENCHMARK_CSV_PATH
    fieldnames = [
        "model", "epoch_best", "val_mIoU", "test_mIoU", "test_mIoU_std",
        "test_mIoU_ci95", "test_acc",
        "IoU_soil", "IoU_bedrock", "IoU_sand", "IoU_bigrock",
        "params_M", "train_time_s"
    ]
    row = {
        "model":          model_name,
        "epoch_best":     round(best_epoch, 1),
        "val_mIoU":       round(val_miou, 4),
        "test_mIoU":      round(test_miou, 4),
        "test_mIoU_std":  round(test_miou_std, 4),
        "test_mIoU_ci95": round(test_miou_ci95, 4),
        "test_acc":       round(test_acc, 4),
        "IoU_soil":       round(iou_soil, 4),
        "IoU_bedrock":    round(iou_bedrock, 4),
        "IoU_sand":       round(iou_sand, 4),
        "IoU_bigrock":    round(iou_bigrock, 4),
        "params_M":       round(params_M, 2),
        "train_time_s":   round(train_time_s, 0),
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ Resultados guardados en {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. VISUALIZACIÓN DE PREDICCIONES
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(model, df_test, device, n=5,
                           save_path="results/predictions.png",
                           img_size=IMG_SIZE):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    CLASS_RGB = {
        0: (230/255, 159/255,   0),
        1: ( 86/255, 180/255, 233/255),
        2: (  0/255, 158/255, 115/255),
        3: (213/255,  94/255,   0),
        255: (0.6, 0.6, 0.6),
    }

    def mask_to_rgb(m):
        rgb = np.zeros((*m.shape, 3))
        for c, col in CLASS_RGB.items():
            rgb[m == c] = col
        return rgb

    model.eval()
    ds = MarsTerrainDataset(
        df_test.sample(n, random_state=42).reset_index(drop=True),
        JointTransformVal(img_size)
    )

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3))
    fig.suptitle("Predicciones vs Ground Truth", fontsize=14, fontweight="bold")

    with torch.no_grad():
        for i in range(n):
            img_t, mask_t, mission = ds[i]
            logits = model(img_t.unsqueeze(0).to(device))
            if isinstance(logits, dict):
                logits = logits["out"]
            pred = logits.argmax(1).squeeze().cpu().numpy()
            gt   = mask_t.numpy()

            mean = np.array(IMAGENET_MEAN); std = np.array(IMAGENET_STD)
            img_np = img_t.permute(1,2,0).numpy()
            img_np = np.clip(img_np * std + mean, 0, 1)

            axes[i][0].imshow(img_np[:,:,0], cmap="gray")
            axes[i][0].set_title(f"{mission} — imagen", fontsize=8)
            axes[i][0].axis("off")

            axes[i][1].imshow(mask_to_rgb(gt))
            axes[i][1].set_title("Ground Truth", fontsize=8)
            axes[i][1].axis("off")

            axes[i][2].imshow(mask_to_rgb(pred))
            axes[i][2].set_title("Predicción", fontsize=8)
            axes[i][2].axis("off")

    patches = [mpatches.Patch(color=CLASS_RGB[c], label=CLASS_NAMES[c])
               for c in range(4)]
    patches.append(mpatches.Patch(color=CLASS_RGB[255], label="ignore"))
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"✅ Predicciones guardadas en {save_path}")


def count_parameters(model: nn.Module) -> float:
    """Retorna número de parámetros en millones."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed()
    df = pd.read_csv("processed/manifest_clean.csv")
    df_train, df_val, df_test = get_or_create_split(df)
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train, df_val, df_test, batch_size=4, num_workers=0
    )
    imgs, masks, missions = next(iter(train_loader))
    print(f"✅ imgs: {imgs.shape}  masks: {masks.shape}")
    print(f"✅ valores únicos en máscara: {masks.unique().tolist()}")
    print("✅ mars_utils.py OK")