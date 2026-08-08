"""微型 YOLO 检测网络 (Phase 1 演示用)。

结构: 3 层量化卷积 backbone + 检测头 (box + objectness)。
输入 64x64 灰度, 输出 16x16 网格, 每格 1 个 anchor:
    box: (cx, cy, w, h) 相对所在 cell, cx/cy 在 [0,1), w/h 以 cell 为单位 (可 >1)
    obj: logit, BCEWithLogits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_conv import QuantConv2d

GRID = 16
CELL = 64 // GRID  # 4px


class MiniYOLO(nn.Module):
    def __init__(self, in_ch=1, quantized=True, quant_w=True):
        super().__init__()
        conv = nn.Conv2d(in_ch, 16, 3, 1, 1)
        self.c1 = QuantConv2d(conv, quant_w=quant_w) if quantized else conv
        conv = nn.Conv2d(16, 16, 3, 2, 1)
        self.c2 = QuantConv2d(conv, quant_w=quant_w) if quantized else conv
        conv = nn.Conv2d(16, 32, 3, 2, 1)
        self.c3 = QuantConv2d(conv, quant_w=quant_w) if quantized else conv
        conv = nn.Conv2d(32, 32, 3, 1, 1)
        self.head_f = QuantConv2d(conv, quant_w=quant_w) if quantized else conv
        self.head_box = nn.Conv2d(32, 4, 1)   # 检测头保持 fp32 (损失信号通道)
        self.head_obj = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        x = F.silu(self.head_f(x))
        box = torch.sigmoid(self.head_box(x))              # B,4,G,G
        box = torch.cat([box[:, :2], box[:, 2:] * 3.0], dim=1)  # w/h 允许超出一个 cell
        obj = self.head_obj(x)                         # B,1,G,G
        return {"box": box, "obj": obj}

    def quantized_convs(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                yield m


def yolo_loss(pred, targets, grid=GRID):
    """targets: list of (cx_px, cy_px, w_px, h_px) per image (pixel 坐标, 已在图内)。

    obj 标签: 目标中心所在 cell = 1; box 标签: 相对 cell 的 (cx, cy, w, h)。
    """
    box = pred["box"]          # B,4,G,G
    obj = pred["obj"]          # B,1,G,G
    B, _, G, _ = box.shape
    obj_t = torch.zeros(B, 1, G, G, device=box.device)
    box_t = torch.zeros(B, 4, G, G, device=box.device)
    mask = torch.zeros(B, G, G, dtype=torch.bool, device=box.device)
    for b, t in enumerate(targets):
        for (cx, cy, w, h) in t:
            ix, iy = int(cx // CELL), int(cy // CELL)
            if not (0 <= ix < G and 0 <= iy < G):
                continue
            obj_t[b, 0, iy, ix] = 1.0
            box_t[b, 0, iy, ix] = cx / CELL - ix
            box_t[b, 1, iy, ix] = cy / CELL - iy
            box_t[b, 2, iy, ix] = w / CELL
            box_t[b, 3, iy, ix] = h / CELL
            mask[b, iy, ix] = True
    obj_loss = F.binary_cross_entropy_with_logits(obj, obj_t)
    box_loss = F.l1_loss(box.permute(0, 2, 3, 1)[mask],
                         box_t.permute(0, 2, 3, 1)[mask]) if mask.any() else torch.tensor(0.0, device=box.device)
    return obj_loss + 2.0 * box_loss


@torch.no_grad()
def detect_metrics(pred, targets, grid=GRID, iou_thr=0.5):
    """简易评估: 每张图取 obj 得分最高的 cell, 与其最匹配目标算 IoU。"""
    obj = pred["obj"].squeeze(1)     # B,G,G
    box = pred["box"]                # B,4,G,G
    hits, total = 0, 0
    for b, t in enumerate(targets):
        total += len(t)
        o = obj[b].flatten()
        if len(t) == 0 or not o.numel():
            continue
        idx = o.argmax()
        ix, iy = idx % GRID, idx // GRID
        cx = (box[b, 0, iy, ix] + ix) * CELL
        cy = (box[b, 1, iy, ix] + iy) * CELL
        w = box[b, 2, iy, ix] * CELL
        h = box[b, 3, iy, ix] * CELL
        best = 0.0
        for (tcx, tcy, tw, th) in t:
            best = max(best, _iou((cx, cy, w, h), (tcx, tcy, tw, th)))
        if best >= iou_thr:
            hits += 1
    return hits / max(total, 1)


def _iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    iw = min(ax1 + a[2], bx1 + b[2]) - max(ax1, bx1)
    ih = min(ay1 + a[3], by1 + b[3]) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)
