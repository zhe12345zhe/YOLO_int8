"""生成 YOLO 格式的合成检测数据集 (Phase 2 用)。

每张图 320x320 RGB: 1-3 个随机矩形, 标签为 YOLO 格式 (class xc yc w h, 归一化)。
输出:
    phase2_ultralytics/data/synth/
        images/train, images/val
        labels/train, labels/val
        data.yaml
"""
import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2

SIZE = 320
ROOT = Path(__file__).resolve().parent / "data" / "synth"


def make_image(rng, size=SIZE):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    bg = int(rng.integers(0, 90))
    img[:] = bg
    labels = []
    n = int(rng.integers(1, 3))
    for _ in range(n):
        w = int(rng.integers(12, 44))
        h = int(rng.integers(12, 44))
        x = int(rng.integers(0, size - w))
        y = int(rng.integers(0, size - h))
        v = int(rng.integers(70, 190))
        color = (v, v, v) if rng.random() < 0.6 else tuple(int(rng.integers(70, 190)) for _ in range(3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cx = (x + w / 2) / size
        cy = (y + h / 2) / size
        nw = w / size
        nh = h / size
        labels.append((0, cx, cy, nw, nh))
    noise = rng.normal(0, 14, (size, size, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, labels


def main():
    random.seed(0)
    np.random.seed(0)
    for split, n in [("train", 600), ("val", 150)]:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(split == "val" and 7 or 0)
        for i in range(n):
            img, labels = make_image(rng)
            name = f"{split}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            with open(lbl_dir / f"{name}.txt", "w") as f:
                for lb in labels:
                    f.write(" ".join(f"{v:.6f}" for v in lb) + "\n")
        print(f"{split}: {n} 张")
    with open(ROOT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(f"""path: {ROOT.as_posix()}
train: images/train
val: images/val
names:
  0: rect
""")
    print("data.yaml 已写入:", ROOT / "data.yaml")


if __name__ == "__main__":
    main()
