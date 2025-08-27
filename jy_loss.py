import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_iou_xywh(a, b, eps=1e-7):
    """
    IoU between boxes given in xywh (center x,y,w,h).
    Used for anchor assignment (size-only IoU by centering at 0,0).
    """
    def xywh2xyxy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    a = xywh2xyxy(a)
    b = xywh2xyxy(b)
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)

    inter_lt = torch.max(a[:, None, :2], b[:, :2])
    inter_rb = torch.min(a[:, None, 2:], b[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = area_a[:, None] + area_b - inter + eps
    return inter / union


class YoloV3Loss(nn.Module):
    """
    YOLOv3-style multi-scale loss (3 heads).
    - Target assignment: choose the best of 9 anchors by (size-only) IoU
    - For each GT, fill (tx,ty,tw,th,obj=1,cls one-hot) at its scale/cell/anchor
    - Loss:
        * obj  : BCE over ALL cells
        * xy   : BCE(sigmoid) over POSITIVE cells only
        * wh   : SmoothL1 over POSITIVE cells only
        * cls  : BCE over POSITIVE cells only
    """
    def __init__(self, anchors, anchor_masks, nc, img_size, strides=(8, 16, 32)):
        super().__init__()
        self.anchors = anchors.clone()          # [9,2] in pixels
        self.anchor_masks = anchor_masks        # tuple of tuples, e.g. ((0,1,2),(3,4,5),(6,7,8))
        self.nc = nc
        self.img_size = img_size
        self.strides = strides
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def build_targets(self, targets, preds_shapes, device):
        """
        targets: [M,6] = (img_idx, class, xc, yc, w, h) normalized to [0,1] wrt img_size
        preds_shapes: list of shapes for 3 scales: (B, na, gh, gw, 5+nc)
        returns: txy, twh, tobj, tcls  (each list length=3, tensors shaped like preds per-scale)
        """
        B = preds_shapes[0][0]
        na = preds_shapes[0][1]
        ghw = [(s[2], s[3]) for s in preds_shapes]  # [(gh_s, gw_s), ...]

        txy, twh, tobj, tcls = [], [], [], []
        for i in range(3):
            gh, gw = ghw[i]
            txy.append(torch.zeros((B, na, gh, gw, 2), device=device))
            twh.append(torch.zeros((B, na, gh, gw, 2), device=device))
            tobj.append(torch.zeros((B, na, gh, gw, 1), device=device))
            tcls.append(torch.zeros((B, na, gh, gw, self.nc), device=device))

        if targets.numel() == 0:
            return txy, twh, tobj, tcls

        # Denormalize to pixels for assignment
        tg = targets.clone()
        tg[:, 2:6] *= self.img_size  # (xc, yc, w, h) in pixels

        # Anchor assignment by size-only IoU
        t_wh = tg[:, 4:6]                      # [N,2]
        a_wh = self.anchors.to(device)         # [9,2]
        t_boxes = torch.cat([torch.zeros_like(t_wh), t_wh], dim=1)  # [N,4] centered at (0,0)
        a_boxes = torch.cat([torch.zeros_like(a_wh), a_wh], dim=1)  # [9,4]
        ious = bbox_iou_xywh(t_boxes, a_boxes)                      # [N,9]
        best_anchor_idx = ious.argmax(1)                            # [N]

        # Fill targets
        for i in range(tg.shape[0]):
            b = int(tg[i, 0].item())        # batch index
            c = int(tg[i, 1].item())        # class id
            a_id = int(best_anchor_idx[i].item())  # 0..8

            # which scale?
            if a_id in self.anchor_masks[0]:
                s = 0
            elif a_id in self.anchor_masks[1]:
                s = 1
            else:
                s = 2
            mask = self.anchor_masks[s]
            a_local = mask.index(a_id)      # 0..2 within the scale

            stride = self.strides[s]
            gh, gw = ghw[s]

            gx = tg[i, 2] / stride
            gy = tg[i, 3] / stride
            gw_ = tg[i, 4] / stride
            gh_ = tg[i, 5] / stride

            gi = int(gx.item())
            gj = int(gy.item())
            if not (0 <= gi < gw and 0 <= gj < gh):
                continue

            # Offsets for xy (relative within cell)
            txy[s][b, a_local, gj, gi, 0] = gx - gi
            txy[s][b, a_local, gj, gi, 1] = gy - gj
            # Log-ratio for wh relative to anchor on this scale
            anchor_wh = (self.anchors[a_id] / stride).to(device)
            twh[s][b, a_local, gj, gi, 0] = torch.log(gw_ / (anchor_wh[0] + 1e-16))
            twh[s][b, a_local, gj, gi, 1] = torch.log(gh_ / (anchor_wh[1] + 1e-16))
            # Objectness + class one-hot
            tobj[s][b, a_local, gj, gi, 0] = 1.0
            tcls[s][b, a_local, gj, gi, c] = 1.0

        return txy, twh, tobj, tcls

    def forward(self, preds, targets):
        """
        preds: list of 3 tensors [B, na, gh, gw, 5+nc]
        targets: [M,6] normalized (img_idx, class, xc, yc, w, h)
        """
        device = preds[0].device
        shapes = [p.shape for p in preds]
        txy, twh, tobj, tcls = self.build_targets(targets, shapes, device)

        loss_xy = torch.tensor(0., device=device)
        loss_wh = torch.tensor(0., device=device)
        loss_obj = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)

        for s in range(3):
            p = preds[s]
            pxy  = p[..., :2]    # raw xy
            pwh  = p[..., 2:4]   # raw wh
            pobj = p[..., 4:5]   # raw obj
            pcls = p[..., 5:]    # raw class logits

            # --- objectness: all cells ---
            loss_obj = loss_obj + self.bce(pobj, tobj[s]).mean()

            # --- positives only for xy/wh/cls ---
            pos = (tobj[s] > 0.5).squeeze(-1)   # [B,na,gh,gw]
            if pos.any():
                loss_xy  = loss_xy  + self.bce(pxy.sigmoid()[pos], txy[s][pos]).mean()
                loss_wh  = loss_wh  + F.smooth_l1_loss(pwh[pos], twh[s][pos], reduction='mean')
                loss_cls = loss_cls + self.bce(pcls[pos], tcls[s][pos]).mean()
            # if no positives, contribute 0 on this scale

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss, (loss_xy.detach(), loss_wh.detach(), loss_obj.detach(), loss_cls.detach())
