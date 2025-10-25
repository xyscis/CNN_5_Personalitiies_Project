import os, sys, math, json, random, pickle
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# -------------------------
# 1) Config
# -------------------------
class Cfg:
    root = "ChaLearn2016_tiny"    # <- change if needed
    train_dir = "train"
    val_dir   = "valid"
    test_dir  = "test"
    ann_dir   = "annotation"
    ann_train = "annotation_training.pkl"
    ann_val   = "annotation_validation.pkl"
    ann_test  = "annotation_test.pkl"

    video_exts = ".mp4"   # adjust if needed
    num_frames = 16          # frames sampled per video
    img_size   = 224
    batch_size = 8
    num_workers = 4
    epochs = 3
    lr = 3e-4
    weight_decay = 1e-4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

traits = ["extraversion","agreeableness","conscientiousness","neuroticism","openness","interview"]

# -------------------------
# 2) Utilities
# -------------------------
def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_ann_dict_keep_names(pkl_path: Path):
    import pickle, numpy as np
    with open(pkl_path, "rb") as f:
        try:
            raw = pickle.load(f)
        except Exception:
            f.seek(0); raw = pickle.load(f, encoding="latin1")

    traits = ["extraversion","agreeableness","conscientiousness","neuroticism","openness","interview"]

    # collect all filenames that appear in any trait
    all_files = set()
    for t in traits:
        if t in raw and isinstance(raw[t], dict):
            all_files.update(raw[t].keys())

    out = {}
    for fname in all_files:
        vec = []
        miss = False
        for t in traits:
            if t in raw and fname in raw[t]:
                vec.append(float(raw[t][fname]))
            else:
                miss = True; break
        if not miss:
            out[str(fname)] = np.array(vec, dtype=np.float32)  # keep 'J4GQm9j0JZ0.003.mp4'
    return out



def extract_uniform_frames(video_path: Path, T: int, resize: int) -> np.ndarray:
    """
    Return [T, H, W, 3] RGB frames uniformly sampled.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, frame_count-1), num=T).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            # fallback: break if something off
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")
    # pad if short
    while len(frames) < T:
        frames.append(frames[-1].copy())
    return np.stack(frames[:T], axis=0)  # [T, H, W, 3]

# -------------------------
# 3) Dataset
# -------------------------
class FirstImpressionsDataset(Dataset):
    def __init__(self, video_dir: Path, ann_map: Dict[str, np.ndarray], T:int, img_size:int, exts:set):
        self.T = T
        self.img_size = img_size

        # Keep the original GT map (keys like 'J4GQm9j0JZ0.003.mp4' -> np.array([6]))
        # Ensure values are float32 tensors later
        self.ann_map = {str(k): v.astype(np.float32) for k, v in ann_map.items()}

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # --- Build tolerant lookup for matching filenames ---
        def norm(s: str) -> str:
            return str(s).strip().lower()

        ann_by_full_lc = {norm(k): k for k in self.ann_map.keys()}
        ann_by_stem = {}
        for k in self.ann_map.keys():
            ann_by_stem[norm(Path(k).stem)] = k

        allowed_exts = {e.lower() for e in (exts | {".webm", ".m4v", ".MP4", ".MOV", ".AVI", ".MKV"})}
        files = [p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]

        self.items = []  # list of (path, ann_key)
        misses = []

        for p in sorted(files):
            fname = p.name
            stem  = p.stem
            f_lc  = norm(fname)
            s_lc  = norm(stem)

            ann_key = None
            if fname in self.ann_map:            # exact match
                ann_key = fname
            elif f_lc in ann_by_full_lc:         # case-insensitive match
                ann_key = ann_by_full_lc[f_lc]
            elif s_lc in ann_by_stem:            # stem-only match
                ann_key = ann_by_stem[s_lc]

            if ann_key is not None:
                self.items.append((p, ann_key))
            else:
                misses.append(fname)

        if not self.items:
            raise RuntimeError(
                f"No videos found (with labels) in {video_dir}\n"
                f"- scanned files: {len(files)}\n"
                f"- example files: {[f.name for f in files[:8]]}\n"
                f"- example ann keys: {list(self.ann_map.keys())[:8]}\n"
                f"- example misses: {misses[:8]}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, ann_key = self.items[idx]
        frames = extract_uniform_frames(path, self.T, self.img_size)         # [T,H,W,3]
        frames = torch.stack([self.tf(fr) for fr in frames], dim=0)          # [T,3,H,W]
        target = torch.from_numpy(self.ann_map[ann_key])                     # [6] float32
        return frames, target, path.name

# -------------------------
# 4) Model
# -------------------------
class VisualRegressor(nn.Module):
    def __init__(self, backbone="resnet50", out_dim=6, pretrained=True):
        super().__init__()
        base = getattr(models, backbone)(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # keep up to global avgpool
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )
        self.out_act = nn.Sigmoid()

    def forward(self, x):  # x: [B,T,3,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.backbone(x).view(B*T, -1)   # [B*T, feat_dim]
        feats = feats.view(B, T, -1).mean(dim=1) # temporal mean pooling
        out = self.head(feats)                   # [B,6]
        return self.out_act(out)

# -------------------------
# 5) Metrics
# -------------------------
def mae(pred, tgt):
    return torch.mean(torch.abs(pred - tgt), dim=0)  # per-dim MAE

def pearsonr(pred, tgt, eps=1e-8):
    # returns per-dim Pearson r
    px = pred - pred.mean(dim=0, keepdim=True)
    tx = tgt  - tgt.mean(dim=0, keepdim=True)
    num = (px*tx).sum(dim=0)
    den = torch.sqrt((px**2).sum(dim=0) * (tx**2).sum(dim=0)) + eps
    return num/den

# -------------------------
# 6) Train/Eval loops
# -------------------------
def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_fn = nn.MSELoss()

    all_mae = []; all_r = []
    total_loss = 0.0; n = 0

    for frames, targets, _ in loader:
        frames = frames.to(device)         # [B,T,3,H,W]
        targets = targets.to(device)       # [B,6]

        with torch.set_grad_enabled(is_train):
            outputs = model(frames)        # [B,6], in [0,1]
            loss = loss_fn(outputs, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * frames.size(0)
        n += frames.size(0)

        all_mae.append(mae(outputs.detach(), targets).cpu())
        all_r.append(pearsonr(outputs.detach(), targets).cpu())

    avg_loss = total_loss / n
    avg_mae = torch.stack(all_mae, dim=0).mean(dim=0)   # [6]
    avg_r   = torch.stack(all_r, dim=0).mean(dim=0)     # [6]
    return avg_loss, avg_mae.numpy(), avg_r.numpy()

# -------------------------
# 7) Main
# -------------------------
def main():
    set_seed(Cfg.seed)
    root = Path(Cfg.root)

    # load annotations per split
    ann_train = load_ann_dict_keep_names(root/ Cfg.ann_dir / Cfg.ann_train)
    ann_val   = load_ann_dict_keep_names(root/ Cfg.ann_dir / Cfg.ann_val)
    ann_test  = load_ann_dict_keep_names(root/ Cfg.ann_dir / Cfg.ann_test)

    # datasets
    ds_tr = FirstImpressionsDataset(root/Cfg.train_dir, ann_train, Cfg.num_frames, Cfg.img_size, Cfg.video_exts)
    ds_va = FirstImpressionsDataset(root/Cfg.val_dir,   ann_val,   Cfg.num_frames, Cfg.img_size, Cfg.video_exts)
    ds_te = FirstImpressionsDataset(root/Cfg.test_dir,  ann_test,  Cfg.num_frames, Cfg.img_size, Cfg.video_exts)

    dl_tr = DataLoader(ds_tr, batch_size=Cfg.batch_size, shuffle=True,  num_workers=Cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=Cfg.batch_size, shuffle=False, num_workers=Cfg.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=Cfg.batch_size, shuffle=False, num_workers=Cfg.num_workers, pin_memory=True)

    # model
    model = VisualRegressor(backbone="resnet50", out_dim=6, pretrained=True).to(Cfg.device)
    # (optional) warmup: freeze backbone for first few epochs
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=Cfg.lr, weight_decay=Cfg.weight_decay)

    best_val = 1e9; best_state = None
    for epoch in range(1, Cfg.epochs+1):
        tr_loss, tr_mae, tr_r = run_epoch(model, dl_tr, optimizer, Cfg.device)

        # unfreeze backbone after 5 epochs
        if epoch == 6:
            for p in model.backbone.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.lr/3, weight_decay=Cfg.weight_decay)

        va_loss, va_mae, va_r = run_epoch(model, dl_va, optimizer=None, device=Cfg.device)

        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(f"       val_MAE per trait (E,A,C,N,O,Int): {np.round(va_mae,4)}")
        print(f"       val_Pearson r   (E,A,C,N,O,Int): {np.round(va_r,4)}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    te_loss, te_mae, te_r = run_epoch(model, dl_te, optimizer=None, device=Cfg.device)
    print("\n=== TEST RESULTS ===")
    print(f"test_loss={te_loss:.4f}")
    print(f"test_MAE (E,A,C,N,O,Int): {np.round(te_mae,4)}")
    print(f"test_r   (E,A,C,N,O,Int): {np.round(te_r,4)}")

if __name__ == "__main__":
    main()
