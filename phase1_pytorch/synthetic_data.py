"""合成检测数据: 64x64 灰度图, 1-3 个随机矩形 (与 YOLO 目标格式一致)。"""
import numpy as np
import torch

SIZE = 64
BRIGHTNESS_RANGE = (100, 200)


def make_sample(rng):
    img = np.zeros((SIZE, SIZE), dtype=np.float32)
    targets = []
    for _ in range(rng.integers(1, 4)):
        w = int(rng.integers(6, 22))
        h = int(rng.integers(6, 22))
        x = int(rng.integers(0, SIZE - w))
        y = int(rng.integers(0, SIZE - h))
        v = float(rng.uniform(*BRIGHTNESS_RANGE))
        img[y:y + h, x:x + w] = v
        targets.append((x + w / 2, y + h / 2, float(w), float(h)))
    return img, targets


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, targets = make_sample(self.rng)
        return torch.from_numpy(img).unsqueeze(0), targets


def get_loaders(batch=32, train_n=3000, val_n=300):
    tr = torch.utils.data.DataLoader(
        SynthDataset(train_n, seed=0), batch_size=batch, shuffle=True,
        num_workers=0, collate_fn=_collate)
    va = torch.utils.data.DataLoader(
        SynthDataset(val_n, seed=123), batch_size=batch, shuffle=False,
        num_workers=0, collate_fn=_collate)
    return tr, va


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tars = [b[1] for b in batch]
    return imgs, tars
