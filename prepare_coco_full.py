"""服务器端脚本: 准备 COCO2017 全量数据集 (train 118287 / val 5000) 的 YOLO 格式。

用途: 在 GPU 租用机上运行, 生成 phase2_ultralytics/datasets/coco-full:
    images/train2017/*.jpg   labels/train2017/*.txt
    images/val2017/*.jpg     labels/val2017/*.txt
    data.yaml (80 类)

下载源:
    images:  https://images.cocodataset.org/zips/train2017.zip  (~19GB)
             https://images.cocodataset.org/zips/val2017.zip    (~780MB)
    label:   https://images.cocodataset.org/annotations/annotations_trainval2017.zip
             (instances_*.json, 转换出完整 YOLO 标签)

用法: python3 prepare_coco_full.py [--keep-zip] [--data-dir /root/autodl-tmp]
      --data-dir 数据集存放根目录 (默认 /root/autodl-tmp, 数据盘; 系统盘只有 30G 放不下 19GB 图片)
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DL = BASE / "coco_download"
DST = BASE / "phase2_ultralytics" / "datasets" / "coco-full"
IMG_URL = "https://images.cocodataset.org/zips/{}.zip"
ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
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


def download(url, dest: Path, desc=""):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  已有 {dest.name} ({dest.stat().st_size/1e6:.0f}MB), 跳过下载")
        return
    print(f"  下载 {desc} -> {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  完成: {dest.name} {dest.stat().st_size/1e6:.0f}MB")


def unzip(zip_path: Path, to: Path):
    print(f"  解压 {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(to)


def convert_annotations(ann_json: Path, img_root: Path, lbl_root: Path, split: str):
    """COCO instance json -> YOLO txt (class cx cy w h 归一化)。"""
    print(f"  转换 {ann_json.name} -> {lbl_root} ...")
    with open(ann_json) as f:
        data = json.load(f)
    cat_map = {c["id"]: i for i, c in enumerate(data["categories"])}
    img_w_h = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    img_file = {im["id"]: im["file_name"] for im in data["images"]}

    by_img = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        x, y, w, h = ann["bbox"]
        W, H = img_w_h[iid]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        by_img.setdefault(iid, []).append(
            f"{cat_map[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    lbl_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for iid, lines in by_img.items():
        stem = img_file[iid].rsplit(".", 1)[0]
        (lbl_root / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    print(f"  写入 {n} 个标签文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--data-dir", type=str, default="/root/autodl-tmp")
    args = ap.parse_args()

    global DL, DST
    DL = Path(args.data_dir) / "coco_download"
    DST = Path(args.data_dir) / "coco-full"
    DL.mkdir(parents=True, exist_ok=True)
    (DL / "images").mkdir(parents=True, exist_ok=True)
    (DL / "ann").mkdir(parents=True, exist_ok=True)

    # ---- 1. 下载或复用 AutoDL 公开数据盘 (COCO2017) 的 zip ----
    #    优先使用 /root/autodl-pub/COCO2017/*.zip (本机拷贝, 秒级), 否则外网下载
    PUB = Path("/root/autodl-pub/COCO2017")
    for split, fname in (("train2017", "train2017.zip"), ("val2017", "val2017.zip")):
        dest = DL / "images" / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  已有 {fname}, 跳过")
            continue
        src = PUB / fname
        if src.exists():
            import shutil as _sh
            print(f"  从公开数据盘拷贝 {fname} ({src.stat().st_size/1e9:.1f}GB) ...")
            _sh.copyfile(src, dest)
            print("  拷贝完成")
            continue
        download(IMG_URL.format(split), dest, split)
    ann_dest = DL / "ann" / "annotations_trainval2017.zip"
    if not (ann_dest.exists() and ann_dest.stat().st_size > 1_000_000):
        pub_ann = PUB / "annotations_trainval2017.zip"
        if pub_ann.exists():
            import shutil as _sh
            print(f"  从公开数据盘拷贝 annotations ...")
            _sh.copyfile(pub_ann, ann_dest)
        else:
            download(ANN_URL, ann_dest, "标注")

    # ---- 2. 解压图片 ----
    for split in ("train2017", "val2017"):
        z = DL / "images" / f"{split}.zip"
        unzip(z, DL / "images" / "unz")
    img_src = {s: DL / "images" / "unz" / s for s in ("train2017", "val2017")}

    # ---- 3. 解压标注并转换 ----
    unzip(DL / "ann" / "annotations_trainval2017.zip", DL / "ann" / "unz")
    ann_dir = DL / "ann" / "unz" / "annotations"

    # ---- 4. 组装 coco-full ----
    for split in ("train2017", "val2017"):
        img_d = DST / "images" / split
        lbl_d = DST / "labels" / split
        img_d.mkdir(parents=True, exist_ok=True)
        for jpg in img_src[split].glob("*.jpg"):
            jpg.rename(img_d / jpg.name)
        convert_annotations(ann_dir / f"instances_{split}.json",
                            img_d, lbl_d, split)
        n_img = len(list(img_d.glob("*.jpg")))
        n_lbl = len(list(lbl_d.glob("*.txt")))
        print(f"[{split}] images={n_img} labels={n_lbl}")

    yaml = DST / "data.yaml"
    yaml.write_text(
        f"path: {DST.resolve()}\n"
        "train: images/train2017\n"
        "val: images/val2017\n"
        "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES)) + "\n",
        encoding="utf-8")
    print(f"data.yaml -> {yaml}")

    if not args.keep_zip:
        shutil.rmtree(DL, ignore_errors=True)
    print("COCO 全量数据集准备完成")


if __name__ == "__main__":
    main()