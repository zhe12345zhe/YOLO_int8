# YOLO int8 量化感知训练 (QAT) —— 反向传播与梯度操作数的 int8 量化

本项目回答：**"用 int8 量化 YOLO 网络反向传播和参数梯度生成过程使用到的操作数"**，并用两个阶段完整实现与验证。

```text
YOLO_int8/
├── phase1_pytorch/          # 阶段 1: 纯 PyTorch 最小原型（讲原理，CPU 可跑）
│   ├── fake_quant.py        #   int8 假量化算子（STE 版 + 真实导数对照版）
│   ├── quant_conv.py        #   量化卷积层：操作数 W_q、A_q 为 int8，反向经 STE
│   ├── mini_yolo.py         #   微型 YOLO 检测网络 + 损失函数
│   ├── synthetic_data.py    #   合成检测数据（64x64 随机矩形）
│   └── train_compare.py     #   FP32 / QAT / TrueGrad / PTQ 对比 + 操作数验证
└── phase2_ultralytics/      # 阶段 2: 真实 ultralytics YOLOv8n 的 QAT 微调
    ├── gen_yolo_data.py     #   YOLO 格式合成数据集（320x320）
    ├── qat_patch.py         #   以"类替换"向 YOLOv8n 插入 int8 假量化
    └── qat_run.py           #   FP32 微调 / PTQ / QAT 对比评估
```

## 核心结论：哪些操作数被 int8 量化

| 环节 | 公式 | 操作数 | 累加 / 结果精度 |
|---|---|---|---|
| 卷积前向 | `Y = W_q ⊗ A_q` | 权重 `W_q`、激活 `A_q` 均为 int8 | int32 累加 → fp32 |
| 权重梯度 | `∂L/∂W = A_q ⊗ E` | 前向保存的 **int8 激活** `A_q` | int32 累加 → fp32 梯度 |
| 输入梯度 | `∂L/∂X = W_q ⊗ E` | **int8 权重** `W_q` | int32 累加 → fp32 |
| 误差信号 | `E = ∂L/∂Y` 穿过 `q` 算子 | 经 STE 直通（fp32 数值） | fp32 |
| 不量化 | Loss 计算（IoU/BCE）、BN、优化器更新、梯度存储 | fp32 | fp32 |

要点：

1. **量化的是"乘法操作数"，不是"结果"**。反向计算 `∂L/∂W` 时，参与矩阵乘法/卷积的因子是前向量化后的 int8 激活 `A_q`；结果（梯度）与累加器始终 int32/fp32，避免误差累积。
2. **STE 与操作数是否量化是两件事**。STE 只是把 `round` 在 `q` 算子处的导数伪造为恒等（`∂q/∂x := 1`），让梯度能穿过量化算子；它不改变"操作数是 int8 值"这一事实。`true`（真实导数）对照实验证明：没有 STE，梯度被 round 杀死，训练停滞。
3. **误差信号 `E` 是否量化是独立决策，且实测无损**。将 `E` 也挂上 int8 假量化（backward hook）后精度几乎不变（见下方"操作数组合消融"）——与论文中 WAGE / DoReFa-Net 将 `E` 量化为 int8 的方向一致。

## 阶段 1 结果（纯 PyTorch 微型 YOLO，40 轮，合成 64x64 数据）

| 方案 | val_loss | hit@0.5 |
|---|---|---|
| FP32 训练（上界） | 0.8245 | 0.420 |
| QAT | 0.8575 | **0.408** |
| PTQ（FP32 权重直接量化） | 1.0858 | 0.361 |
| TrueGrad（无 STE 对照） | 1.1930 | 0.107 ← 训练停滞 |

**反向传播操作数验证**（QAT 模型，c1 层）：
- (a) STE 恒等：`x` 的梯度 == `convT(E, W_q)`，相对误差 **0.00e+00**
- (b) 权重梯度操作数：`w.grad == unfold(A_q) × E`，相对误差 **1.82e-05** —— autograd 计算 `∂L/∂W` 时使用的确实是 int8 激活 `A_q`
- (c) 权重 int8 量化误差：**0.39%**

## 操作数组合消融（W / A / E 三个 int8 开关，15 轮，微型 YOLO）

`train_modes.py` 统一控制三个操作数是否走 int8 假量化（W=权重、A=前向量化激活、E=反向误差梯度），证明每块量化各自的作用：

| 消融（量化开关） | hit@0.5 | 相对 FP32 |
|---|---|---|
| FP32（全不量化） | 0.4427 | 上界 |
| W+A+E（全量化） | 0.4392 | -0.0035 |
| W+A（E 不量化，QAT 现行） | 0.4393 | -0.0034 |
| **A+E（权重保持 FP32）** | **0.4397** | **-0.0030（最佳）** |
| 仅 E | 0.4376 | -0.0051 |

结论：
- **权重不量化反而是最佳组合**（0.4397）：对微型 YOLO 权重 int8 约束贡献有限，去掉它还能提速并减少对 ΔW 的失真；
- **E 量化接近无损**：W+A+E 0.4392 vs W+A 0.4393 仅掉 0.0001（噪声内）；`grad_quant.py` 独立复验 E-int8 0.4392 ≈ E-fp32 0.4393 —— 与反向传播中 E 只被量化后的 A 相乘的理论一致；
- **A 的量化是精度损失的真正来源**：仅量化 E 掉 0.0051，说明激活区分度主要在 A。

复现：`python train_modes.py`（约 4 分钟，5 组合一次跑完）。

## 阶段 2 结果（ultralytics YOLOv8n，12 轮，合成 320x320 数据）

| 方案 | mAP50 | mAP50-95 |
|---|---|---|
| FP32 微调（上界） | 0.9839 | 0.9136 |
| PTQ（FP32 权重 + 静态校准，不训练） | 0.9837（掉 0.0002） | 0.8786（**掉 0.0350**） |
| QAT（量化模拟下微调） | 0.9825（掉 0.0014） | **0.8822**（掉 0.0315） |

## 阶段 2 结果（真实数据集 COCO128，50 轮，320px，CPU 约 40 分钟）

| 方案 | mAP50 | mAP50-95 |
|---|---|---|
| FP32 微调（上界） | 0.6835 | 0.5290 |
| PTQ（FP32 权重 + 静态校准，不训练） | 0.6766（掉 0.0069） | 0.5089（**掉 0.0201**） |
| QAT（量化模拟下微调） | 0.6969（**+0.0134**） | **0.5281**（掉 0.0009，几乎无损） |

真实数据集上 QAT 的效果清晰可见：PTQ 在 mAP50-95 上掉 2 个百分点，QAT 微调后几乎零损失（且 mAP50 还略高于 FP32），证明训练中经 STE 的量化噪声适配（权重对 int8 表示"免疫"）有效。

## QAT 真的在量化吗？——四层硬证据

担心 QAT 结果"几乎无损"是因为量化退化成 FP32 微调？以下证据逐层排除（全部可复现）：

| # | 证据 | 结果 |
|---|---|---|
| [1] 结构 | 训练结束后检查网络里的卷积类型（探针保存的训练态模型 `out/patched_model.pt`） | **QConv2d × 45**（每层含权重量化 qw + 激活量化 qa）；best.pt 保存时被 ultralytics `strip_optimizer` 从 yaml 重建回 plain Conv2d——这正是"看文件像 FP32"的假象来源 |
| [2] 运行计数 | 训练循环内埋点统计 `quantize_int8` 调用次数（每步 45 层 × 2 张量 = 90） | 探针训练每轮 **1440 次 / 平均每 batch 90 次**；若补丁失效则为 0 |
| [3] 消融 | 同一权重，量化开关（`QUANT_ENABLED`）两种模式评估 | 见下方 2×3 矩阵：量化开启与关闭的 mAP 不同 → 推理确实走了量化路径 |
| [4] 网格 | 权重 int8 量化误差 `max\|W−W_q\|/max\|W\|` | QAT 权重 0.3910~0.3920% vs FP32 权重 0.3920%（QAT 权重更贴近 int8 网格，STE 拉近效果） |

[3] 消融矩阵（COCO128，320px，静态校准后评估，`qat_prove.py`）：

| 权重来源 | 量化开启（mAP50 / mAP50-95） | 量化关闭（mAP50 / mAP50-95） |
|---|---|---|
| QAT 50 轮 | 0.6876 / **0.5283** | 0.6919 / 0.5297 |
| FP32 50 轮 | 0.6842 / 0.5190 | 0.6835 / 0.5290 |

注意行内差异：**量化开关对同一权重结果不同**（QAT mAP50-95 掉 0.14%、FP32 掉 1.0%）→ 量化真实参与了推理；而列间对比 QAT-量化后 0.5283 反而高于 FP32-量化后 0.5190 → 正是 QAT 把权重训练得"对 int8 免疫"的证据。若只是 FP32 微调，QAT 权重量化后应与 FP32 权重同样掉点（mAP50-95 掉 ~1%）。

复现：`python qat_probe_train.py --epochs 3 --imgsz 192`（探针+计数，约 3 分钟）→ `python qat_prove.py`（结构/消融/网格，约 6 分钟）。

## 复现方法

```bash
# 阶段 1（约 5 分钟，CPU）
pip install torch torchvision ultralytics
cd phase1_pytorch
python train_compare.py            # 完整（40/40/8 epochs）
python train_compare.py --quick   # 快速（15/15/4 epochs）

# 阶段 2（约 12 分钟，CPU；yolov8n.pt 与 coco128 首次运行自动下载到 datasets/）
cd phase2_ultralytics
python gen_yolo_data.py                          # 生成合成数据集
python qat_run.py --epochs 12 --imgsz 256 --batch 8   # 合成数据完整
python qat_run.py --quick                            # 合成数据快速（3 epochs, 192px）
python qat_run.py --data coco128 --epochs 50 --imgsz 320 --batch 16  # 真实数据 COCO128
```

## 关键实现细节

1. **fake_quant.py**：前向 `x_q = (clamp(round(x/s + zp), -128, 127) - zp) * s`；反向 STE `dy · 1{x 在量化范围内}`；对照版 `_FakeQuantizeTrue` 返回 0 梯度。
2. **权重量化** per-output-channel 对称；**激活量化** per-tensor 对称，训练中动态取整/静态校准切换（`ActQuant.static` 与 `calib_max`）。
3. **qat_patch.py**：`m.__class__ = QConv2d` 类替换不改 state_dict 键名，可直接加载预训练权重；检测头（Detect）保持 fp32；必须用 `on_train_start` 回调打补丁（trainer 会在 setup 阶段从 checkpoint 重建模型）；PTQ 评估用训练集静态校准冻结激活范围。
4. **失误避坑**：`model.export(format="tflite", int8=True)` 是训练后量化（PTQ）而非 QAT——它只做导出校准，模型参数在训练中从未"见过"量化噪声。真正的 QAT 必须在训练循环里插入假量化算子，让反向梯度（经 STE）与量化算子交互，这正是本项目实现的内容。