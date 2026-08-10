#!/usr/bin/env bash
# GPU 远程部署 + 训练脚本 (在 Linux GPU 机上运行)
#
# 用法:
#   模式 1 (COCO-big 子集, 本机已有 tar):
#     bash deploy_remote.sh big /path/to/coco_big_dataset.tar.gz [epochs=15] [imgsz=320] [batch=16]
#   模式 2 (COCO2017 全量 118k, 服务器下载):
#     bash deploy_remote.sh full [epochs=15] [imgsz=320] [batch=16]
#
# 通用说明:
#   - 需要项目代码已上传: phase1_pytorch/ phase2_ultralytics/(qat_run_big.py 等)
#   - 三阶段 (FP32 / QAT_WA / QAT_WAE) 依次运行, 每阶段完成后自动续跑下一阶段
#     某阶段中断后重跑本脚本, 会自动 resume 该阶段 (last.pt 存在)

set -e
MODE="$1"                       # big | full
EPOCHS="${2:-15}"
IMGSZ="${3:-320}"
BATCH="${4:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=== [0] 检查依赖 (torch+ultralytics) ==="
python3 - <<'PY'
import sys
try:
    import torch, ultralytics
    ok = torch.cuda.is_available()
    if ok:
        print(f"torch {torch.__version__} / ultralytics {ultralytics.__version__} / "
              f"GPU {torch.cuda.get_device_name(0)}")
    else:
        print("警告: CUDA 不可用! 请安装 CUDA 版 torch: "
              "pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先: pip install ultralytics onnx, 并按 pytorch.org 安装 cuda 版 torch")
    sys.exit(1)
PY

if [ "$MODE" = "big" ]; then
    TAR_PATH="$2"
    [ -z "$TAR_PATH" ] && { echo "模式 big 需要数据集 tar 路径"; exit 1; }
    echo "=== [1] 解压数据集 $TAR_PATH ==="
    mkdir -p "${SCRIPT_DIR}/phase2_ultralytics/datasets"
    tar -xzf "$TAR_PATH" -C "${SCRIPT_DIR}/phase2_ultralytics/datasets"
    DATA_YAML="${SCRIPT_DIR}/phase2_ultralytics/datasets/coco-big/data.yaml"
    python3 - "$DATA_YAML" <<'PY'
import sys
from pathlib import Path
y = Path(sys.argv[1])
d = y.parent
lines = [f"path: {d}" if l.startswith("path:") else l for l in y.read_text().splitlines()]
y.write_text("\n".join(lines) + "\n")
print(f"data.yaml 已修正 -> {y}")
PY
elif [ "$MODE" = "full" ]; then
    DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"   # 数据盘, 可环境变量覆盖
    echo "=== [1] 下载并准备 COCO2017 全量 (118k/5k, ~20GB) -> $DATA_ROOT ==="
    if [ ! -f "${DATA_ROOT}/coco-full/data.yaml" ]; then
        python3 "${SCRIPT_DIR}/prepare_coco_full.py" --data-dir "${DATA_ROOT}"
    else
        echo "  coco-full 已存在, 跳过准备"
    fi
    DATA_YAML="${DATA_ROOT}/coco-full/data.yaml"
else
    echo "用法: bash deploy_remote.sh big <tar> [epochs] [imgsz] [batch] | full [epochs] [imgsz] [batch]"
    exit 1
fi

PROJ_DIR="${SCRIPT_DIR}/proj"
echo "=== [2] 依次训练: FP32 -> QAT(W+A) -> QAT(W+A+E) ==="
cd "${PROJ_DIR}/phase2_ultralytics"
for stage in fp32 wa wae; do
    echo "---- stage=${stage} $(date '+%F %T') ----"
    if python3 qat_run_big.py --data "${DATA_YAML}" --stage "${stage}" \
        --epochs "${EPOCHS}" --imgsz "${IMGSZ}" --batch "${BATCH}" \
        > "${LOG_DIR}/stage_${stage}.log" 2>&1; then
        echo "stage=${stage} 完成"
    else
        echo "stage=${stage} 退出码 $? (日志: logs/stage_${stage}.log)"
    fi
done

echo "=== 结果汇总 (out/phase2_big_results.txt) ==="
cat "${PROJ_DIR}/phase2_ultralytics/out/phase2_big_results.txt" 2>/dev/null || true
echo ""
echo "=== 全部完成 $(date '+%F %T') ==="