"""从 COCO2017 val（5000 张真实图）构造更大规模的 YOLO 检测数据集。

拆分: train 1200 张 / val 500 张 (固定 seed, 与 COCO128 的 128 张形成规模对照)。
输出: datasets/coco-big/{images,labels}/{train,val} + data.yaml
"""
import random
from pathlib import Path

SRC_IMG = Path("datasets/_dl/val2017/coco/images/val2017")
SRC_LBL = Path("datasets/_dl/val2017/coco/labels/val2017")
DST = Path("datasets/coco-big")
N_TRAIN, N_VAL = 1200, 500
SEED = 42

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main():
    lbl_files = sorted(Path(SRC_LBL).glob("*.txt"))
    valid = []
    for lbl in lbl_files:
        img = SRC_IMG / (lbl.stem + ".jpg")
        if img.exists() and lbl.stat().st_size > 0:
            valid.append(lbl)
    print(f"有标签且图片存在的样本: {len(valid)}")

    rng = random.Random(SEED)
    rng.shuffle(valid)
    splits = [("train", valid[:N_TRAIN]), ("val", valid[N_TRAIN:N_TRAIN + N_VAL])]

    for split, items in splits:
        img_d = DST / "images" / split
        lbl_d = DST / "labels" / split
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)
        for lbl in items:
            img = SRC_IMG / (lbl.stem + ".jpg")
            img.replace(img_d / img.name)      # 移动而非复制, 省空间
            lbl.rename(lbl_d / lbl.name)
        print(f"[{split}] {len(items)} 张")

    yaml_path = DST / "data.yaml"
    yaml_path.write_text(
        f"path: {DST.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES)) + "\n",
        encoding="utf-8",
    )
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()