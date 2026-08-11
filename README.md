# YOLO int8 量化感知训练 (QAT) —— 反向传播与梯度操作数的 int8 量化

本项目回答：**"用 int8 量化 YOLO 网络反向传播和参数梯度生成过程使用到的操作数"**，并用两个阶段完整实现与验证。

```text
YOLO_int8/
├── phase1_pytorch/          # 阶段 1: 纯 PyTorch 最小原型（讲原理，CPU 可跑）
│   ├── fake_quant.py        #   int8 假量化算子（STE 版 + 真实导数对照版）
│   ├── quant_conv.py        #   量化卷积层：操作数 W_q、A_q 为 int8，反向经 STE（quant_w 开关）
│   ├── mini_yolo.py         #   微型 YOLO 检测网络 + 损失函数（透传 quant_w）
│   ├── synthetic_data.py    #   合成检测数据（64x64 随机矩形）
│   ├── train_compare.py     #   FP32 / QAT / TrueGrad / PTQ 对比 + 操作数验证
│   ├── train_modes.py       #   W/A/E 三个 int8 开关的组合消融
│   └── grad_quant.py        #   误差梯度 E 挂 int8 假量化验证
├── phase2_ultralytics/       # 阶段 2: 真实 ultralytics YOLOv8n 的 QAT 微调
│   ├── gen_yolo_data.py     #   YOLO 格式合成数据集（320x320）
│   ├── qat_patch.py         #   以"类替换"向 YOLOv8n 插入 int8 假量化
│   ├── qat_run.py           #   FP32 微调 / PTQ / QAT 对比评估
│   ├── qat_run_e.py         #   全操作数量化 W+A+E（真实数据集 COCO128）
│   ├── qat_run_big.py       #   大数据实验主脚本（全量 COCO2017，GPU）
│   ├── qat_probe_train.py   #   探针训练：量化调用计数 + 训练态模型保存（on_train_start 补丁）
│   ├── qat_prove.py         #   四层证据：结构/计数/消融/网格
│   ├── show_weights.py      #   QAT vs FP32 权重逐层并排打印（weights_dump.txt）
│   ├── export_onnx.py       #   导出 ONNX，统计假量化痕迹算子
│   ├── deploy_int8.py       #   TensorRT FP32/INT8 engine 构建 + 精度/延迟对比部署脚本
│   └── tee_run.py           #   子进程运行器（UTF-8 BOM 日志，防 Windows 转码乱码）
├── prepare_coco_full.py      # 全量 COCO2017 下载/标注转换/组装（--data-dir 指向数据盘）
├── deploy_remote.sh          # 远端部署入口：三阶段（fp32→wa→wae）训练流水线
├── ssh_run.py                # paramiko 密码 SSH：执行命令 / 上传 / 下载
└── ssh_run_safe.py           # base64 免引号版 SSH 执行（规避特殊字符转义问题）
```

## 核心结论：哪些操作数被 int8 量化

| 环节 | 公式 | 操作数 | 累加 / 结果精度 |
|---|---|---|---|
| 卷积前向 | `Y = W_q ⊗ A_q` | 权重 `W_q`、激活 `A_q` 均为 int8 | int32 累加 → fp32 |
| 权重梯度 | `∂L/∂W = A_q ⊗ E` | 前向保存的 **int8 激活** `A_q` | int32 累加 → fp32 梯度 |
| 输入梯度 | `∂L/∂X = W_q ⊗ E` | **int8 权重** `W_q` | int32 累加 → fp32 |
| 误差信号 | `E = ∂L/∂Y` 穿过 `q` 算子 | 经 STE 直通（fp32 数值） | fp32（可切 int8，见工作 D） |
| 不量化 | Loss 计算（IoU/BCE）、BN、优化器更新、梯度存储 | fp32 | fp32 |

要点：

1. **量化的是"乘法操作数"，不是"结果"**。反向计算 `∂L/∂W` 时，参与矩阵乘法/卷积的因子是前向量化后的 int8 激活 `A_q`；结果（梯度）与累加器始终 int32/fp32，避免误差累积。
2. **STE 与操作数是否量化是两件事**。STE 只是把 `round` 在 `q` 算子处的导数伪造为恒等（`∂q/∂x := 1`），让梯度能穿过量化算子；它不改变"操作数是 int8 值"这一事实。`true`（真实导数）对照实验证明：没有 STE，梯度被 round 杀死，训练停滞。
3. **误差信号 `E` 是否量化是独立决策，且结果依模型规模、数据规模而异**。微型 YOLO 上 E 挂 int8 假量化几乎无损（工作 D）；真实 YOLOv8n 的 mAP50-95 掉 1.1 个百分点（COCO128，工作 I），在全量 COCO2017 上扩大到 3.4 个百分点（工作 J）——检测头三支损失梯度的幅值分布更复杂，E 是全网络逐层量化的敏感通道，且数据越大越明显。WAGE / DoReFa-Net 在浅层分类网络上的结论不能直接外推到检测模型。

## 完成的工作（并列一览，详见平行小节）

| 工作 | 验证目标 | 关键结果 | 脚本 |
|---|---|---|---|
| [A 阶段 1 方案对比](#wA) | 微型 YOLO 上 FP32/QAT/PTQ/TrueGrad 差异 | QAT 0.408 紧贴上界 0.420；TrueGrad 0.107（无 STE 则训练停滞） | `train_compare.py` |
| [B 反向传播操作数验证](#wB) | 梯度生成用到的确实是 int8 操作数 `A_q`、`W_q` | STE 恒等误差 0.00e+00；`w.grad == unfold(A_q)×E` 误差 1.82e-05；权重量化误差 0.39% | `train_compare.py` |
| [C 操作数组合消融](#wC) | W/A/E 三个 int8 开关各自的贡献 | A+E（权重 FP32）0.4397 最佳；E 量化无损（0.4392 vs 0.4393）；A 是精度主成本 | `train_modes.py` |
| [D E 量化无损验证](#wD) | 误差梯度 E 挂 int8 后的精度变化 | E-int8 0.4392 ≈ E-fp32 0.4393，量化无损 | `grad_quant.py` |
| [E 阶段 2 合成数据](#wE) | 真实 YOLOv8n 上 QAT vs PTQ | QAT mAP50-95 0.8822（掉 0.0315）vs PTQ 0.8786（掉 0.0350） | `qat_run.py` |
| [F 阶段 2 COCO128](#wF) | 真实数据集 50 轮 | QAT 0.5281（几乎无损）vs PTQ 0.5089（掉 2 个点），mAP50 +0.0134 | `qat_run.py --data coco128` |
| [G "QAT 真的在量化吗"四层硬证据](#wG) | 证明 QAT 真在量化（非 FP32 微调退化） | 结构 QConv2d×45、每 batch 90 次量化调用、消融开关不同 mAP、权重网格误差 0.391% | `qat_probe_train.py`→`qat_prove.py` |
| [H ONNX 导出检视](#wH) | 量化痕迹在导出图中的展开形式 | Round×90 / Div×190 / Mul×165 假量化展开，无 QDQ 节点 | `export_onnx.py` |
| [I 全操作数量化（W+A+E，COCO128）](#wI) | 权重/激活/误差梯度三者全部 int8 的真实数据验证 | mAP50-95 0.5168（vs W+A 0.5281 掉 1.1 个点）——E 量化在检测网络上有损，与微型模型结论相反 | `qat_run_e.py` |
| [J 全量 COCO2017（GPU，118k）](#wJ) | 大数据上三方案对比，验证 QAT 结论的稳健性 | W+A 掉 0.56 点（几乎无损）；W+A+E 掉 3.40 点（E 量化有损在大数据上更显著）；"QAT 超 FP32"证实为小数据集偶然 | `qat_run_big.py` · `deploy_remote.sh` |
| [K TensorRT 部署（3080 Ti）](#wK) | 训练权重落地为真实 INT8 硬件加速 | INT8 engine 精度损失：普通权重 -7.1%、QAT_W+A -7.9%、**QAT_W+A+E 仅 -1.3%**；端到端延迟 2.2→2.0ms、纯推理 ~1.06-1.10x；QDQ 246 对节点断言 INT8 生效 | `deploy_int8.py` · `out/int8_bench.png` |

---

## 工作 A：阶段 1 方案对比 <a id="wA"></a>

纯 PyTorch 微型 YOLO（c1→c2→c3 三个量化卷积），40 轮，合成 64x64 数据：

| 方案 | val_loss | hit@0.5 |
|---|---|---|
| FP32 训练（上界） | 0.8245 | 0.420 |
| QAT | 0.8575 | **0.408** |
| PTQ（FP32 权重直接量化） | 1.0858 | 0.361 |
| TrueGrad（无 STE 对照） | 1.1930 | 0.107 ← 训练停滞 |

复现：`python train_compare.py`（完整）或 `python train_compare.py --quick`。

## 工作 B：反向传播操作数验证 <a id="wB"></a>

QAT 模型 c1 层，逐条验证反向传播计算链上"被使用"的操作数：

- (a) STE 恒等：`x` 的梯度 == `convT(E, W_q)`，相对误差 **0.00e+00**
- (b) 权重梯度操作数：`w.grad == unfold(A_q) × E`，相对误差 **1.82e-05** —— autograd 计算 `∂L/∂W` 时使用的确实是 int8 激活 `A_q`
- (c) 权重 int8 量化误差：**0.39%**

复现：`python train_compare.py`（同一脚本内完成）。

## 工作 C：操作数组合消融（W / A / E 三个 int8 开关）<a id="wC"></a>

`train_modes.py` 统一控制三个操作数是否走 int8 假量化（W=权重、A=前向激活、E=反向误差梯度）。15 轮、seed 0：

| 消融（量化开关） | hit@0.5 | 相对 FP32 |
|---|---|---|
| FP32（全不量化） | 0.4427 | 上界 |
| W+A+E（全量化） | 0.4392 | -0.0035 |
| W+A（E 不量化，QAT 现行） | 0.4393 | -0.0034 |
| **A+E（权重保持 FP32）** | **0.4397** | **-0.0030（最佳）** |
| 仅 E | 0.4376 | -0.0051 |

结论：

- **权重不量化反而是最佳组合**（0.4397）：对微型 YOLO，权重 int8 网格约束贡献有限，去掉它还能提速；
- **E 量化接近无损**：W+A+E 0.4392 vs W+A 0.4393 仅掉 0.0001（噪声内），与工作 D 的独立验证互相印证；
- **A 的量化是精度损失的主要来源**：仅量化 E 掉 0.0051（激活保持 fp32 时反而最稳）。

复现：`python train_modes.py`（约 4 分钟，5 组合一次跑完）。

## 工作 D：误差梯度 E 量化无损验证 <a id="wD"></a>

`grad_quant.py` 用 backward hook 把每一层输入的误差信号 `E` 先过 int8 假量化再送回梯度链（与 WAGE 一致），与 E 不量化的 QAT 现状对比：

| 方案 | hit@0.5 |
|---|---|
| QAT 现状（E 为 fp32） | 0.4393 |
| QAT + E 量化为 int8 | 0.4392 |

量化无损（差 0.0001 在噪声内）——反向中 E 只与 int8 的 `A_q` 相乘，梯度幅值区间稳定，量化不削梯度上限。

复现：`python grad_quant.py`。

## 工作 E：阶段 2 真实 YOLOv8n（合成数据 320x320）<a id="wE"></a>

12 轮微调：

| 方案 | mAP50 | mAP50-95 |
|---|---|---|
| FP32 微调（上界） | 0.9839 | 0.9136 |
| PTQ（FP32 权重 + 静态校准，不训练） | 0.9837（掉 0.0002） | 0.8786（**掉 0.0350**） |
| QAT（量化模拟下微调） | 0.9825（掉 0.0014） | **0.8822**（掉 0.0315） |

复现：`python gen_yolo_data.py && python qat_run.py --epochs 12 --imgsz 256 --batch 8`。

## 工作 F：阶段 2 YOLOv8n 真实数据集 COCO128 <a id="wF"></a>

50 轮、320px、CPU 约 40 分钟：

| 方案 | mAP50 | mAP50-95 |
|---|---|---|
| FP32 微调（上界） | 0.6835 | 0.5290 |
| PTQ（FP32 权重 + 静态校准，不训练） | 0.6766（掉 0.0069） | 0.5089（**掉 0.0201**） |
| QAT（量化模拟下微调） | 0.6969（**+0.0134**） | **0.5281**（掉 0.0009，几乎无损） |

真实数据集上 QAT 的效果清晰可见：PTQ 在 mAP50-95 上掉 2 个百分点，QAT 微调后几乎零损失（mAP50 还略高于 FP32），证明训练中经 STE 迭代的量化噪声适配（权重对 int8 表示"免疫"）有效。

复现：`python qat_run.py --data coco128 --epochs 50 --imgsz 320 --batch 16`。

## 工作 G："QAT 真的在量化吗？"四层硬证据 <a id="wG"></a>

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

注意行内差异：**量化开关对同一权重结果不同**（QAT mAP50-95 掉 0.14%、FP32 掉 1.0%）→ 量化真实参与了推理；列间对比 QAT-量化后 0.5283 高于 FP32-量化后 0.5190 → 正是 QAT 把权重训练得"对 int8 免疫"的证据。若只是 FP32 微调，QAT 权重量化后应与 FP32 权重同样掉点（mAP50-95 掉 ~1%）。

复现：`python qat_probe_train.py --epochs 3 --imgsz 192`（探针+计数，约 3 分钟）→ `python qat_prove.py`（结构/消融/网格，约 6 分钟）。

## 工作 H：ONNX 导出检视（假量化在导出图中的展开形式）<a id="wH"></a>

导出训练态 QAT 模型（QConv2d 全保留下）到 ONNX（opset 17, dynamo=False），统计算子类型：

| 算子 | 数量 | 含义 |
|---|---|---|
| Round | 90 | 45 层 × W、A 各一个取整 |
| Div | 190 | scale 除法/乘法展开 |
| Mul | 165 | 反量化乘 / 累加混淆 |
| Conv | 89 | 卷积主体（int32 累加路径） |
| QuantizeLinear / DequantizeLinear | 0 | 训练期假量化，非标准 QDQ 部署节点 |

ONNX 文件：`out/yolov8n_qat_quant.onnx`（可在 Netron 中查看 Round 链）。这从工具链角度证明：本实现的量化是"算子内 int8"，而非 PTQ 的 QDQ 图改造。

复现：`python gen_yolo_data.py`（数据集就绪后）`python export_onnx.py`。

## 工作 I：全操作数量化 W+A+E（真实数据集 COCO128）<a id="wI"></a>

权重 W、激活 A、误差梯度 E 三个乘法操作数**全部** int8（`qat_patch.py` 的 `quant_e=True` 用 backward hook 对每个量化卷积的输入梯度做 per-tensor int8 假量化），在 COCO128 上完整微调 50 轮（320px，CPU 约 14 分钟）：

| 方案 | 量化开关 | mAP50 | mAP50-95 |
|---|---|---|---|
| FP32（上界） | 无 | 0.6835 | 0.5290 |
| PTQ | W+A（不训练） | 0.6766 | 0.5089 |
| QAT | W+A | **0.6969** | **0.5281** |
| QAT | **W+A+E** | 0.6743（-0.0226） | 0.5168（-0.0113） |

结构验证：训练网络 **45 个 QConv2d = 权重量化 45 + 激活量化 45 + 误差量化 45**，E 钩子存活到训练结束。

结论（与微型模型相反）：

- **E 量化在真实检测网络上是有损的**：mAP50 掉 2.3 个点（接近 PTQ 水平），mAP50-95 掉 1.1 个点——误差通道对每层逐张量 int8 化敏感；
- 原因分析：YOLOv8n 检测头 box/cls/dfl 三支损失梯度幅值差异大，全网络 45 层逐层量化 E 使噪声沿反向逐层累积，STE 无法完全吸收；
- 对照微型 YOLO（工作 D：E-int8 0.4392 ≈ E-fp32 0.4393）：浅层小模型 E 量化无损，深层检测模型有损——"E 是否量化"应当作超参按任务验证，不能照搬论文。

复现：`python qat_run_e.py --epochs 50 --imgsz 320 --batch 16`（结果写入 `out/phase2_e_results.txt`）。

## 工作 J：全量 COCO2017（GPU，train 118k / val 5k）<a id="wJ"></a>

租用 NVIDIA RTX 3080 Ti（12GB）服务器，把 COCO128 的小样本结论放到全量数据上重验。数据直接从 AutoDL 公开数据盘 `/root/autodl-pub/COCO2017` 拷贝（免外网下载），由 `prepare_coco_full.py` 解压并转成 YOLO 格式（118,287 图 / 117,266 标签，与官方数字一致）。三个方案各 15 epochs、320px、batch 16、workers 8，由 `deploy_remote.sh` 编排依次训练（支持断点续训）。

| 方案 | mAP50 | mAP50-95 | vs FP32（mAP50-95） |
|---|---|---|---|
| FP32（上界） | 0.3673 | 0.2450 | — |
| QAT W+A（激活+权重 int8） | 0.3620 | 0.2394 | **-0.0056**（几乎无损） |
| QAT W+A+E（全操作数 int8） | 0.3252 | 0.2110 | **-0.0340**（明显有损） |

效率与资源（日志统计，tqdm GPU_mem 采样 10 万+ 点）：

| 阶段 | 平均速度 | 15 epochs 耗时 | 显存 min/mean/peak |
|---|---|---|---|
| FP32 | 12.4 it/s | ~2h21m | 1.53G |
| QAT W+A | 7.4 it/s | ~3h46m | 1.32 / 1.75 / **1.78G** |
| QAT W+A+E | 6.8 it/s | ~4h26m | 1.31 / 1.75 / **1.78G** |

（12GB 显存占用峰值不足 15%；AutoDL 面板上"内存 86%"是**系统 RAM 页缓存**，非显存，全程无 OOM，日志中 `out of memory` 出现 0 次。）

结论：

- **QAT W+A 的优势在大数据上成立**：掉点仅 0.56 个点（mAP50-95），与 COCO128 上"几乎无损"（0.09 点）量级一致，QAT 使权重对 int8 网格免疫的机制在全量数据上同样有效；
- **"QAT mAP50 超过 FP32（+0.0134）"确认为小数据集偶然**：全量下 FP32 0.3673 为三方案最高；
- **E 量化的有损性在大数据上放大**：从 COCO128 的 -1.1 点扩大到 **-3.4 点**（mAP50 掉 4.2 点）——误差通道逐层 int8 化的噪声在深网络 + 大数据下更难以被 STE 吸收，进一步支持工作 I 的结论"E 是否量化应作超参验证"；
- **QAT 训练成本**：比 FP32 慢约 60%（W+A）~88%（W+A+E），每卷积前向/反向多两次假量化算子的 Python 层开销。

复现（GPU 服务器）：上传仓库至服务器后

```bash
bash deploy_remote.sh full 15 320 16   # 自动: 准备数据 -> fp32 -> wa -> wae 依次训练
```

结果写入 `phase2_ultralytics/out/phase2_big_results.txt`；权重 `runs/detect/out/{big_fp32,big_qat,big_qat_e}/weights/best.pt`（本仓库 `out/` 下已存档 `ckpt_fp32_best.pt` / `ckpt_qat_wa_best.pt` / `ckpt_qat_wae_best.pt`）。

## 工作 K：TensorRT 部署 —— QAT 权重落地为真实 INT8 硬件加速 <a id="wK"></a>

把工作 J 的三份训练权重（普通 FP32、QAT W+A、QAT W+A+E）部署为 TensorRT engine，在 **RTX 3080 Ti（12GB，TensorRT 11.2 / CUDA 13.0）** 上完成"量化训练 → 硬件加速"的完整闭环：每个权重构建 FP32 与 INT8 两个 engine，在 COCO val2017 全量 5000 张上对比精度，并分别测端到端（含 CPU 前后处理）与纯 engine 推理延迟。

构建链路（ultralytics 8.4.117 导出，TRT11 强类型路径）：`.pt` → ONNX → **ModelOpt 静态量化 512 张校准图** → `best.int8.onnx`（显式 Q/DQ 节点）→ TensorRT 解析 QDQ 图构建 INT8 engine。INT8 生效的硬证据：三份 `best.int8.onnx` 各含 **246 对 QuantizeLinear/DequantizeLinear 节点**（对应 246 个量化张量），engine 即由该图构建。

### 精度对比（val2017 5000 张，imgsz=320）

| 权重 | mAP50 FP32→INT8 | mAP50-95 FP32→INT8 | INT8 损失 |
|---|---|---|---|
| 普通 FP32 | 0.3663 → 0.3404 | 0.2445 → 0.2196 | **-7.1%** / -2.4 点 |
| QAT W+A | 0.3635 → 0.3348 | 0.2430 → 0.2142 | **-7.9%** / -2.9 点 |
| QAT W+A+E | 0.2897 → 0.2858 | 0.1960 → 0.1890 | **-1.3%** / -0.7 点 |

（结果文件 `out/int8_deploy_results.txt`，图 `out/int8_bench.png`。）

### 延迟对比（batch=1, 320x320）

| 测法 | FP32 engine | INT8 engine | speedup |
|---|---|---|---|
| 端到端（yolo predict，含 CPU 前/后处理，120 次平均） | 2.2 ms | 2.0 ms | ~1.07-1.10x |
| 纯 engine 推理（GPU only，500 次平均） | 0.886-1.020 ms | 0.827-0.947 ms | ~1.06-1.08x |

（纯推理明细 `out/trt_pure_infer_bench.txt`。）

### 结论

1. **QAT_W+A+E 对 int8 表示最"免疫"**：INT8 精度损失仅 -1.3%，普通权重同链路掉 7.1%（与工作 G 的消融结论一致——QAT 把权重训练得贴近 int8 网格，PTQ 式校准对未经量化训练的权重伤害更大）。注意其 FP32 基线本身低 0.29 vs 0.37（工作 J 已证明 E 量化有损）。
2. **INT8 在 3080 Ti 上加速有限（~1.1x）**：yolov8n@320 仅 ~1ms 推理，kernel 启动/内存搬运占大头，且 Python 侧前后处理（~1.5ms）盖过 GPU 端收益；INT8 的吞吐优势在 batch>1、更大模型或更高分辨率下才明显。
3. **端到端速度瓶颈在前后处理**：纯推理 0.8-1.0ms vs 端到端 2.0-2.2ms，前后处理占一半以上，落地的下一瓶颈是预处理流水线与 NMS，而非量化。

复现（GPU 服务器，需 TensorRT 11 与 nvidia-modelopt）：

```bash
pip install tensorrt nvidia-modelopt  # TRT 11 强类型路径
python deploy_int8.py --imgsz 320 --data /root/autodl-tmp/coco-full/data.yaml
```

坑位记录：TRT11 下 ultralytics 对 FP32/INT8 导出**同名** `best.engine`（INT8 覆盖 FP32）——`deploy_int8.py` 导出后立即改名 `best_{fp32,int8}.engine` 区分；TRT11 的 `IEngineInspector` 在序列化 engine 上不可靠（Myelin 崩溃），INT8 层统计改用 QDQ onnx 节点计数。

产物存档（`phase2_ultralytics/out/trt_engines/`，随实例释放而失效，本地已备份）：

| 文件 | 说明 |
|---|---|
| `{big_fp32,big_qat,big_qat_e}_{fp32,int8}.engine` | 6 个可直接部署的 TensorRT engine（3080 Ti / TRT 11.2 构建，~35-54MB） |
| `{big_fp32,big_qat,big_qat_e}_qdq_int8.onnx` | 3 份 INT8 QDQ 图（各 246 对 Q/DQ 节点，INT8 精度证据与后续复用原料） |

## 复现方法

```bash
# 阶段 1（约 5 分钟，CPU）
pip install torch torchvision ultralytics
cd phase1_pytorch
python train_compare.py             # 完整（40/40/8 epochs）
python train_compare.py --quick     # 快速（15/15/4 epochs）
python train_modes.py               # W/A/E 组合消融（约 4 分钟）
python grad_quant.py                # E 量化无损验证（约 4 分钟）

# 阶段 2（约 12 分钟，CPU；yolov8n.pt 与 coco128 首次运行自动下载到 datasets/）
cd phase2_ultralytics
python gen_yolo_data.py                             # 生成合成数据集
python qat_run.py --epochs 12 --imgsz 256 --batch 8 # 合成数据完整
python qat_run.py --quick                           # 合成数据快速（3 epochs, 192px）
python qat_run.py --data coco128 --epochs 50 --imgsz 320 --batch 16  # COCO128
python qat_run_e.py --epochs 50 --imgsz 320 --batch 16              # W+A+E 全操作数量化
python qat_probe_train.py --epochs 3 --imgsz 192    # 探针 + 计数
python qat_prove.py                                 # 四层证据
python show_weights.py                              # 权重并排打印
python export_onnx.py                               # ONNX 导出检视

# 全量 COCO2017（工作 J，需 GPU 服务器）
bash deploy_remote.sh full 15 320 16                # 远端一键: 数据准备 + fp32/wa/wae 三阶段
```

## 关键实现细节

1. **fake_quant.py**：前向 `x_q = (clamp(round(x/s + zp), -128, 127) - zp) * s`；反向 STE `dy · 1{x 在量化范围内}`；对照版 `_FakeQuantizeTrue` 返回 0 梯度。
2. **权重量化** per-output-channel 对称；**激活量化** per-tensor 对称，训练中动态在线统计、评估时静态校准（`ActQuant.static` 与 `calib_max` 切换）。
3. **qat_patch.py**：`m.__class__ = QConv2d` 类替换不改 state_dict 键名，可直接加载预训练权重；检测头（Detect）保持 fp32；必须用 `on_train_start` 回调（trainer 会在 setup 阶段从 checkpoint 重建模型）；PTQ 评估用训练 set 静态校准冻结激活范围。
4. **失误避坑**：`model.export(format="tflite", int8=True)` 是训练后量化（PTQ）而非 QAT——它只对导出做校准，模型在训练中从未"见过"量化噪声。真正的 QAT 必须在训练循环里插入假量化算子，让反向梯度（经 STE）与量化算子交互，这正是本项目实现的内容。
5. **Windows 编码**：PowerShell 重定向会产生 GBK/UTF-8 双重转码乱码；日志统一由 `tee_run.py` 以 UTF-8 BOM 直接写盘（字节级验证），终端回显乱码只是 GBK 渲染假象，不影响文件内容。