# train.py
import os
# ===== GPU 환경설정은 torch import 전에 두는 게 가장 안전합니다 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                 # 0번 GPU만 사용 (원하면 변경)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import yaml
import math
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from dataset import YoloTxtDataset, collate_fn
from model import yolo_v3
from loss import YoloV3Loss
from tqdm import tqdm

# ---- 간단 EMA ----
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + (1. - self.decay) * msd[k].detach())

# ---- EarlyStopping ----
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.best = float('inf')
        self.count = 0
        self.min_delta = min_delta
        self.stop = False
    def step(self, val):
        if val < self.best - self.min_delta:
            self.best = val
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

# GPU 연산 최적화
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True   # Ampere 이상 TF32
torch.cuda.empty_cache()
try:
    torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 0번 GPU 메모리 최대 사용
except Exception:
    pass

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='dataset.yaml')
    ap.add_argument('--epochs', type=int, default=100, help='최종 목표 에포크 (resume 시 start_epoch부터 epochs-1까지 수행)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', type=str, default='', help='비워두면 자동 선택, 혹은 "0","1"...')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', type=str, default='', help='체크포인트 경로 (e.g., runs/exp0/weights/last.pt)')
    ap.add_argument('--expdir', type=str, default='runs/exp0', help='실험 디렉터리')
    ap.add_argument('--clip', type=float, default=10.0, help='grad clip max-norm')
    ap.add_argument('--patience', type=int, default=6, help='LR plateau patience')
    ap.add_argument('--early', type=int, default=10, help='early stopping patience')
    return ap.parse_args()

def main():
    args = parse_args()
    # ===== device 선택 =====
    device = torch.device(
        f'cuda:{args.device}' if (args.device != '' and torch.cuda.is_available())
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    print('device:', device)

    # ===== 데이터셋 =====
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg['names']
    nc = len(names)
    img_size = args.img

    train_set = YoloTxtDataset(args.data, 'train', img_size=img_size)
    val_set   = YoloTxtDataset(args.data, 'val',   img_size=img_size)
    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=(args.workers > 0)
    )

    # ===== 모델/옵티마이저/스케일러/로스/스케줄러/EMA/ES =====
    model = yolo_v3(nc=nc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 최신 권고 API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6, verbose=True
    )
    ema = ModelEMA(model, decay=0.9999, device=device)
    early = EarlyStopping(patience=args.early, min_delta=0.0)
    criterion = YoloV3Loss(model.anchors, model.anchor_masks, nc, img_size, strides=model.strides)

    # ===== 실험 디렉터리 =====
    exp_dir = args.expdir
    weights_dir = os.path.join(exp_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # ====== RESUME ======
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        # 1) 모델
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=True)
        else:  # state_dict-only
            model.load_state_dict(ckpt, strict=True)
            ckpt = {}  # 이후 키 접근 방지
        # 2) EMA
        if 'ema' in ckpt and ckpt['ema'] is not None:
            try:
                ema.ema.load_state_dict(ckpt['ema'], strict=False)
            except Exception as e:
                print(f'[warn] EMA state not loaded: {e}')
        # 3) 옵티마/스케줄러/스케일러
        if 'optimizer' in ckpt:
            try: optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e: print(f'[warn] optimizer state not loaded: {e}')
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            try: scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e: print(f'[warn] scheduler state not loaded: {e}')
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            try: scaler.load_state_dict(ckpt['scaler'])
            except Exception as e: print(f'[warn] scaler state not loaded: {e}')
        # 4) 에폭/베스트
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_loss = float(ckpt.get('best_loss', best_loss))
        print(f'[resume] from {args.resume} at epoch {start_epoch-1}, best_loss={best_loss:.4f}')

    # ====== 학습 루프 ======
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        parts_accum = torch.zeros(4, dtype=torch.float32, device=device)  # xy, wh, obj, cls
        num_batches = 0

        for imgs, targets, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs)
                loss, parts = criterion(preds, targets)  # parts: (lxy,lwh,lobj,lcls)

            scaler.scale(loss).backward()
            # clip (unscale 후)
            scaler.unscale_(optimizer)
            if args.clip and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running += loss.item()
            parts_accum += torch.stack(parts).detach()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, num_batches)
        parts_mean = (parts_accum / max(1, num_batches)).tolist()  # 4개 float

        # ===== Validation =====
        model.eval()
        val_running = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_running += loss.item()
        val_loss = val_running / max(1, len(val_loader))
        print(f"[epoch {epoch+1}] parts(xy/wh/obj/cls)={parts_mean} | train={train_loss:.4f} val={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        # ===== 스케줄러/세이브/얼리스탑 =====
        scheduler.step(val_loss)

        # 저장(모든 상태 포함)
        ckpt = {
            'epoch': epoch,
            'best_loss': best_loss,
            'model': model.state_dict(),
            'ema': ema.ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'names': names,
            'img_size': img_size,
            'anchors': model.anchors.detach().cpu().numpy().tolist(),
            'anchor_masks': [list(m) for m in model.anchor_masks]
        }
        torch.save(ckpt, os.path.join(weights_dir, 'last.pt'))

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, os.path.join(weights_dir, 'best.pt'))
            # state-only (EMA 기준)
            torch.save(ema.ema.state_dict(), os.path.join(weights_dir, 'best_state_dict.pt'))

        # Early stopping
        early.step(val_loss)
        if early.stop:
            print(f"[EarlyStop] validation loss plateaued (patience={args.early}).")
            break

    print("Done. Best val loss:", best_loss)

if __name__ == '__main__':
    main()
