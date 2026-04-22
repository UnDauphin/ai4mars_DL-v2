"""
mars_utils_fast.py
==================
Versión optimizada de mars_utils.py para RTX 3050 laptop (4GB VRAM).

Cambios respecto a mars_utils.py:
  1. Asume imágenes ya prerredimensionadas a 256×256 (sin TF.resize en transforms)
  2. AMP (Automatic Mixed Precision) integrado correctamente — scaler creado UNA vez
  3. torch.compile() opcional para PyTorch >= 2.0
  4. Subset estratificado para reducir dataset manteniendo representatividad
  5. num_workers=4 + persistent_workers=True por defecto
  6. batch_size=16 (AMP permite duplicar el batch sin OOM en 4GB)

Uso:
    from mars_utils_fast import *
    df = pd.read_csv("processed/manifest_256.csv")   # ← manifest prerredimensionado
"""

import os, random, pickle, time, csv
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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 4
IGNORE_INDEX  = 255
SEED          = 42
IMG_SIZE      = 256
BATCH_SIZE    = 16          # AMP permite 16 en 4GB VRAM

CLASS_NAMES   = {0:"soil", 1:"bedrock", 2:"sand", 3:"big_rock"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PROCESSED_DIR   = Path("processed")
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR     = Path("results")
for d in [PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SPLIT_INDICES_PATH = PROCESSED_DIR / "split_indices.pkl"
BENCHMARK_CSV_PATH = RESULTS_DIR   / "benchmark_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS — SIN resize (ya están prerredimensionadas)
# ─────────────────────────────────────────────────────────────────────────────
class JointTransformTrain:
    """
    Augmentaciones para train. SIN resize porque las imágenes
    ya están en 256×256 (prerredimensionadas con preprocess_resize.py).
    """
    def __init__(self):
        self.color_jit = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Flip horizontal
        if random.random() > 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        # Flip vertical
        if random.random() > 0.5:
            img = TF.vflip(img); mask = TF.vflip(mask)
        # Rotación ±15°
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST,  fill=IGNORE_INDEX)
        # Color jitter (solo imagen)
        img = self.color_jit(img)
        # ToTensor + Normalize
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask

class JointTransformVal:
    """Solo normalize. Sin resize ni augmentaciones."""
    def __init__(self):
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    def __call__(self, img, mask):
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class MarsTerrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask, row["mission"]


# ─────────────────────────────────────────────────────────────────────────────
# SUBSET ESTRATIFICADO (reduce dataset manteniendo representatividad)
# ─────────────────────────────────────────────────────────────────────────────
def make_subset(df, n_per_mission, seed=SEED):
    """
    Toma n_per_mission imágenes por misión.
    Recomendado para entrenamiento rápido:
      n_per_mission=700 → ~2100 train total, ~60s/epoch en RTX 3050
    """
    return (
        df.groupby("mission", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), n_per_mission), random_state=seed))
          .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def get_or_create_split(df, train_ratio=0.70, val_ratio=0.15,
                        seed=SEED, save_path=SPLIT_INDICES_PATH):
    if save_path.exists():
        with open(save_path,"rb") as f: indices = pickle.load(f)
        df_train = df.iloc[indices["train"]].reset_index(drop=True)
        df_val   = df.iloc[indices["val"]  ].reset_index(drop=True)
        df_test  = df.iloc[indices["test"] ].reset_index(drop=True)
        print(f"✅ Split cargado desde {save_path}")
    else:
        train_f, val_f, test_f = [], [], []
        for _, group in df.groupby("mission"):
            g = group.sample(frac=1, random_state=seed)
            n = len(g); n_tr = int(n*train_ratio); n_va = int(n*val_ratio)
            train_f.append(g.iloc[:n_tr])
            val_f.append(  g.iloc[n_tr:n_tr+n_va])
            test_f.append( g.iloc[n_tr+n_va:])
        df_train = pd.concat(train_f).sample(frac=1,random_state=seed).reset_index(drop=True)
        df_val   = pd.concat(val_f  ).sample(frac=1,random_state=seed).reset_index(drop=True)
        df_test  = pd.concat(test_f ).sample(frac=1,random_state=seed).reset_index(drop=True)
        assert len(set(df_train["id"]) & set(df_test["id"])) == 0, "Data leakage!"
        indices = {"train": df_train.index.tolist(),
                   "val":   df_val.index.tolist(),
                   "test":  df_test.index.tolist()}
        with open(save_path,"wb") as f: pickle.dump(indices, f)
        print(f"✅ Split generado y guardado en {save_path}")

    print(f"Train:{len(df_train):>6} {df_train['mission'].value_counts().to_dict()}")
    print(f"Val  :{len(df_val):>6} {df_val['mission'].value_counts().to_dict()}")
    print(f"Test :{len(df_test):>6} {df_test['mission'].value_counts().to_dict()}")
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(df):
    counts  = df["mission"].value_counts().to_dict()
    weights = torch.DoubleTensor(df["mission"].map(lambda m: 1.0/counts[m]).values)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADERS — optimizados para Windows + RTX 3050
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(df_train, df_val, df_test,
                      batch_size=BATCH_SIZE, num_workers=4,
                      balance_missions=True):
    """
    num_workers=4 funciona bien en Windows con 12 hilos lógicos.
    persistent_workers=True evita reiniciar workers entre epochs.
    pin_memory=True acelera transferencia CPU→GPU.
    """
    pw = num_workers > 0
    train_ds = MarsTerrainDataset(df_train, JointTransformTrain())
    val_ds   = MarsTerrainDataset(df_val,   JointTransformVal())
    test_ds  = MarsTerrainDataset(df_test,  JointTransformVal())

    sampler = make_weighted_sampler(df_train) if balance_missions else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
class MetricsAccumulator:
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.correct = 0; self.total = 0

    def update(self, preds, targets):
        if preds.dim() == 4: preds = preds.argmax(dim=1)
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        v = t != self.ignore_index
        p, t = p[v], t[v]
        self.correct += int((p==t).sum()); self.total += int(v.sum())
        for c in range(self.num_classes):
            self.tp[c] += int(((p==c)&(t==c)).sum())
            self.fp[c] += int(((p==c)&(t!=c)).sum())
            self.fn[c] += int(((p!=c)&(t==c)).sum())

    def compute(self):
        iou = {}
        for c in range(self.num_classes):
            d = self.tp[c]+self.fp[c]+self.fn[c]
            iou[CLASS_NAMES[c]] = float(self.tp[c]/d) if d>0 else 0.0
        return {"mIoU": float(np.mean(list(iou.values()))),
                "IoU_per_class": iou,
                "pixel_accuracy": self.correct/max(self.total,1)}


# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes=num_classes; self.ignore_index=ignore_index; self.smooth=smooth
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        dice=0.0; n=0
        for c in range(self.num_classes):
            t=(targets[valid]==c).float()
            if t.sum()==0: continue
            p=probs[:,c][valid]
            inter=(p*t).sum()
            dice+=(2*inter+self.smooth)/(p.sum()+t.sum()+self.smooth); n+=1
        return 1.0 - dice/max(n,1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.alpha=alpha; self.gamma=gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        fl = self.alpha*(1-pt)**self.gamma*ce
        return fl[targets != self.ignore_index].mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)
    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP — AMP con scaler creado UNA sola vez (no dentro del loop)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, aux_weight=0.0):
    """
    scaler: GradScaler creado fuera del loop de epochs (no recrear cada epoch).
    """
    model.train()
    total_loss = 0.0
    acc = MetricsAccumulator()

    for imgs, masks, _ in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # más rápido que zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(imgs)
                logits = output["out"] if isinstance(output, dict) else output
                loss   = criterion(logits, masks)
                if aux_weight > 0 and isinstance(output, dict) and "aux" in output:
                    loss = loss + aux_weight * criterion(output["aux"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(imgs)
            logits = output["out"] if isinstance(output, dict) else output
            loss   = criterion(logits, masks)
            if aux_weight > 0 and isinstance(output, dict) and "aux" in output:
                loss = loss + aux_weight * criterion(output["aux"], masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        acc.update(logits.detach(), masks)

    r = acc.compute(); r["loss"] = total_loss/len(loader)
    return r


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    acc = MetricsAccumulator()
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(imgs)
                logits = output["out"] if isinstance(output, dict) else output
                loss   = criterion(logits, masks)
        else:
            output = model(imgs)
            logits = output["out"] if isinstance(output, dict) else output
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        acc.update(logits, masks)
    r = acc.compute(); r["loss"] = total_loss/len(loader)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL — scaler creado aquí una vez para todo el entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, model_name="model",
                num_epochs=30, patience=10, aux_weight=0.0,
                use_amp=True):

    use_amp = use_amp and (device != "cpu")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    best_state = None; best_miou = -1.0; no_improve = 0
    history = {"train":[], "val":[]}
    t_start = time.time()

    print(f"\n{'='*55}")
    print(f"  {model_name} | device={device} | AMP={use_amp}")
    print(f"{'='*55}")

    best_epoch = 1
    for epoch in range(1, num_epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                             device, scaler=scaler, aux_weight=aux_weight)
        va = evaluate(model, val_loader, criterion, device, scaler=scaler)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["mIoU"])
            else:
                scheduler.step()

        history["train"].append(tr); history["val"].append(va)
        print(f"Ep {epoch:3d}/{num_epochs} | "
              f"loss={tr['loss']:.4f} mIoU={tr['mIoU']:.4f} | "
              f"val={va['mIoU']:.4f} | "
              f"rock={va['IoU_per_class']['big_rock']:.4f}")

        if va["mIoU"] > best_miou:
            best_miou  = va["mIoU"]; best_epoch = epoch; no_improve = 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"epoch":epoch,"model_state":best_state,"val_metrics":va},
                       CHECKPOINTS_DIR/f"{model_name}_best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping en epoch {epoch}"); break

    train_time = time.time() - t_start
    print(f"\n  Mejor val mIoU={best_miou:.4f} (epoch {best_epoch}) | {train_time:.0f}s")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CHECKPOINTS_DIR/f"{model_name}_final_state_dict.pth")
    return history, best_miou, best_epoch, train_time


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RUN (3 seeds)
# ─────────────────────────────────────────────────────────────────────────────
def run_multi_seed(model_fn, df_train, df_val, df_test,
                   criterion_fn, optimizer_fn, scheduler_fn,
                   model_name, device,
                   seeds=[42,123,7], num_epochs=30, patience=10,
                   batch_size=BATCH_SIZE, num_workers=4,
                   aux_weight=0.0, use_amp=True,
                   n_per_mission=None):
    """
    n_per_mission: si se especifica, usa subset estratificado de ese tamaño.
    Recomendado: n_per_mission=700 para RTX 3050 (→ ~2100 train, ~60s/epoch).
    """
    all_test = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'─'*50}\n  Seed {seed} | {model_name}")

        # Subset opcional
        tr = make_subset(df_train, n_per_mission) if n_per_mission else df_train
        va = make_subset(df_val,   max(n_per_mission//4, 50)) if n_per_mission else df_val

        train_loader, val_loader, test_loader = build_dataloaders(
            tr, va, df_test, batch_size=batch_size, num_workers=num_workers
        )

        model     = model_fn().to(device)
        criterion = criterion_fn()
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        history, best_miou, best_epoch, train_time = train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, model_name=f"{model_name}_seed{seed}",
            num_epochs=num_epochs, patience=patience,
            aux_weight=aux_weight, use_amp=use_amp,
        )

        test_res = evaluate(model, test_loader, criterion, device)
        test_res.update({"seed":seed,"best_epoch":best_epoch,"train_time":train_time})
        all_test.append(test_res)
        print(f"  Test mIoU={test_res['mIoU']:.4f}")

    miou_vals = [r["mIoU"] for r in all_test]
    summary = {
        "model":model_name,
        "mIoU_mean":float(np.mean(miou_vals)),
        "mIoU_std": float(np.std(miou_vals)),
        "mIoU_ci95":float(1.96*np.std(miou_vals)/np.sqrt(len(miou_vals))),
        "pixel_acc_mean":float(np.mean([r["pixel_accuracy"] for r in all_test])),
        "train_time_mean":float(np.mean([r["train_time"] for r in all_test])),
        "best_epoch_mean":float(np.mean([r["best_epoch"] for r in all_test])),
        "per_seed":all_test,
    }
    for c in CLASS_NAMES.values():
        vals = [r["IoU_per_class"][c] for r in all_test]
        summary[f"iou_{c}_mean"] = float(np.mean(vals))
        summary[f"iou_{c}_std"]  = float(np.std(vals))

    print(f"\n{'='*55}")
    print(f"  {model_name}: mIoU={summary['mIoU_mean']:.4f}±{summary['mIoU_std']:.4f}"
          f" IC95=±{summary['mIoU_ci95']:.4f}")
    for c in CLASS_NAMES.values():
        print(f"  IoU({c})={summary[f'iou_{c}_mean']:.4f}±{summary[f'iou_{c}_std']:.4f}")
    print(f"{'='*55}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO CSV
# ─────────────────────────────────────────────────────────────────────────────
def append_benchmark_results(model_name, best_epoch, val_miou, test_miou,
                             test_miou_std, test_miou_ci95, test_acc,
                             iou_soil, iou_bedrock, iou_sand, iou_bigrock,
                             params_M, train_time_s):
    fieldnames = ["model","epoch_best","val_mIoU","test_mIoU","test_mIoU_std",
                  "test_mIoU_ci95","test_acc","IoU_soil","IoU_bedrock","IoU_sand",
                  "IoU_bigrock","params_M","train_time_s"]
    row = dict(zip(fieldnames,[model_name,round(best_epoch,1),round(val_miou,4),
                               round(test_miou,4),round(test_miou_std,4),
                               round(test_miou_ci95,4),round(test_acc,4),
                               round(iou_soil,4),round(iou_bedrock,4),
                               round(iou_sand,4),round(iou_bigrock,4),
                               round(params_M,2),round(train_time_s,0)]))
    write_header = not BENCHMARK_CSV_PATH.exists()
    with open(BENCHMARK_CSV_PATH,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"✅ Resultados en {BENCHMARK_CSV_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def visualize_predictions(model, df_test, device, n=5,
                           save_path="results/predictions.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    CLASS_RGB = {0:(230/255,159/255,0),1:(86/255,180/255,233/255),
                 2:(0,158/255,115/255),3:(213/255,94/255,0),255:(.6,.6,.6)}
    def m2rgb(m):
        rgb=np.zeros((*m.shape,3))
        for c,col in CLASS_RGB.items(): rgb[m==c]=col
        return rgb
    model.eval()
    ds = MarsTerrainDataset(
        df_test.sample(n,random_state=42).reset_index(drop=True),
        JointTransformVal()
    )
    fig,axes=plt.subplots(n,3,figsize=(12,n*3))
    fig.suptitle("Predicciones vs GT",fontsize=14,fontweight="bold")
    with torch.no_grad():
        for i in range(n):
            img_t,mask_t,mission=ds[i]
            out=model(img_t.unsqueeze(0).to(device))
            logits=out["out"] if isinstance(out,dict) else out
            pred=logits.argmax(1).squeeze().cpu().numpy()
            gt=mask_t.numpy()
            mean=np.array(IMAGENET_MEAN); std=np.array(IMAGENET_STD)
            img_np=np.clip(img_t.permute(1,2,0).numpy()*std+mean,0,1)
            axes[i][0].imshow(img_np[:,:,0],cmap="gray"); axes[i][0].set_title(f"{mission}"); axes[i][0].axis("off")
            axes[i][1].imshow(m2rgb(gt));   axes[i][1].set_title("GT");   axes[i][1].axis("off")
            axes[i][2].imshow(m2rgb(pred)); axes[i][2].set_title("Pred"); axes[i][2].axis("off")
    patches=[mpatches.Patch(color=CLASS_RGB[c],label=CLASS_NAMES[c]) for c in range(4)]
    patches.append(mpatches.Patch(color=CLASS_RGB[255],label="ignore"))
    fig.legend(handles=patches,loc="lower center",ncol=5,fontsize=9,bbox_to_anchor=(.5,-.01))
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(save_path,dpi=130,bbox_inches="tight"); plt.show()
    print(f"✅ {save_path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())/1e6
