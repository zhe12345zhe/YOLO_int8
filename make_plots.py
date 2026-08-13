"""生成训练曲线图（COCO2017 与 VisDrone），供 README 使用。

数据源: D:\\a\\AI_project\\YOLO_int8_Drive\\dl_1\\proj\\phase2_ultralytics\\runs\\detect\\out
  - big_fp32   results.csv 缺失(未保存), 图中以最终值水平虚线标注
  - big_qat / big_qat_e      COCO2017 主实验 15 epochs (big_qat_e 取前 15)
  - vis_fp32 / vis_wa / vis_wae   VisDrone 微调 15 epochs
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\a\AI_project\YOLO_int8_Drive\dl_1\proj\phase2_ultralytics\runs\detect\out"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

COLORS = {"FP32": "#1F3864", "QAT W+A": "#ED7D31", "QAT W+A+E": "#C00000"}
STYLES = {"FP32": "--", "QAT W+A": "-", "QAT W+A+E": "-"}


def load(name):
    with open(os.path.join(BASE, name, "results.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0].keys()}


def plot_pair(fig_name, title, series, n_epochs, fp32_final=None, path_suffix=""):
    """series: [(tag, data), ...]"""
    ep = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, mkey in zip(axes, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], ["mAP50", "mAP50-95"]):
        for tag, d in series:
            ax.plot(ep, d[key][:n_epochs], marker="o", ms=4,
                    color=COLORS[tag], linestyle=STYLES[tag], label=tag)
        if fp32_final is not None:
            ax.axhline(fp32_final[0] if key.endswith("mAP50(B)") else fp32_final[1],
                       color=COLORS["FP32"], linestyle="--", lw=1.2)
            ax.text(0.5, 1.02, "FP32 最终值(15ep)",
                    transform=ax.transAxes, ha="center", fontsize=9, color=COLORS["FP32"])
        ax.set_title(mkey, fontsize=13)
        ax.set_xlabel("epoch"); ax.set_ylabel(mkey)
        ax.set_xticks(list(ep)); ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, fig_name), dpi=150)
    plt.close(fig)


def plot_loss(fig_name, title, series, n_epochs):
    ep = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, lkey in zip(axes, ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
                             ["box loss", "cls loss", "dfl loss"]):
        for tag, d in series:
            ax.plot(ep, d[key][:n_epochs], marker="o", ms=4,
                    color=COLORS[tag], linestyle=STYLES[tag], label=tag)
        ax.set_title("train " + lkey, fontsize=13)
        ax.set_xlabel("epoch"); ax.set_ylabel(lkey)
        ax.set_xticks(list(ep)); ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, fig_name), dpi=150)
    plt.close(fig)


# ---- COCO2017（工作 J）：big_qat / big_qat_e，FP32 以最终值标注 ----
coco_qat = load("big_qat")
coco_qat_e = load("big_qat_e")
n = 15
plot_pair("coco2017_map.png",
          "COCO2017 全量训练：QAT W+A / QAT W+A+E（yolov8n, 320px, bs16）",
          [("QAT W+A", coco_qat), ("QAT W+A+E", coco_qat_e)], n,
          fp32_final=(0.3673, 0.2450))
plot_loss("coco2017_loss.png",
          "COCO2017 全量训练：损失曲线", [("QAT W+A", coco_qat), ("QAT W+A+E", coco_qat_e)], n)

# ---- VisDrone（工作 M）：三方案 ----
vis = {t: load(nm) for nm, t in [("vis_fp32", "FP32"), ("vis_wa", "QAT W+A"), ("vis_wae", "QAT W+A+E")]}
plot_pair("visdrone_map.png",
          "VisDrone 微调：FP32 / QAT W+A / QAT W+A+E（yolov8n, 640px, bs16, 15 epochs）",
          [("FP32", vis["FP32"]), ("QAT W+A", vis["QAT W+A"]), ("QAT W+A+E", vis["QAT W+A+E"])], 15)
plot_loss("visdrone_loss.png",
          "VisDrone 微调：训练损失曲线",
          [("FP32", vis["FP32"]), ("QAT W+A", vis["QAT W+A"]), ("QAT W+A+E", vis["QAT W+A+E"])], 15)

print("saved:", sorted(os.listdir(OUT)))
