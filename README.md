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
│   ├── int8_engine.py       #   INT8 训练引擎（方案 B）：cuDNN INT8 fprop + SwitchBack dW（gemmEx 免转置）
│   ├── int8_engine_train.py #   INT8 引擎训练/吞吐/数值验证入口（ultralytics 集成）
│   ├── lt_ex.cu             #   cublasGemmEx int8 扩展（dW 转置版 GEMM，torch cpp_extension 编译）
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
| [L INT8 训练引擎（方案 B，GPU）](#wL) | 真 int8 GEMM 训练链路（前向 cuDNN INT8 + SwitchBack dW）在 3080 Ti 上能否加速 | 数值正确（前向/dW 误差 ~1.2%，dX 精确）；三轮工程 0.34x→0.59x 仍逊 fp32；大模型(yolov8l/x 0.41-0.49x)翻不了盘——1x1 层 int8 慢 3 倍、dX/dgrad 在 sm86 无 INT8、CUDA 13.1 无 EPILOGUE_SCALE；瓶颈为 **Python 调度 35ms**；**AMP 对照：大模型 1.6-1.9x 且不掉点（官方默认配置），一行代码胜三轮工程** | `int8_engine.py` · `int8_engine_train.py` · `lt_ex.cu` |

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

## 工作 L：INT8 训练引擎（方案 B）—— 反向真正落盘到 int8 硬件 GEMM <a id="wL"></a>

前面工作（A-J）的量化都是**假量化**（数值模拟，int8 仅在计算图中以 round/scale 形式出现），TensorRT 部署（工作 K）只覆盖推理。本工作把训练侧反传真正落到 int8 硬件矩阵乘法上，回答：**"用真 int8 GEMM 算子跑训练（前向 + SwitchBack dW），在 3080 Ti 上能否为 yolov8n 加速"**。

### 方案 B 设计（SwitchBack 式）

| 环节 | 精度策略 | 最终实现（三轮工程迭代后） |
|---|---|---|
| 卷积前向 | **int8** | **cuDNN 9 图 API INT8 隐式 GEMM 卷积**（免 im2col，sm86 实测大层快 2-4x）；x per-tensor 动态量化（在线 absmax） |
| 权重梯度 dW | **int8（SwitchBack）** | 自写 CUDA extension 调 `cublasGemmEx`（transb=T 免内存转置）：`xq²ᵈᵀ @ dyq`，`(sx·sdy)` 放缩 |
| 输入梯度 dX | **fp32（高精度）** | `conv_transpose2d`，不做 int8（cuDNN wgrad/dgrad 的 INT8 在 sm86 无引擎） |

量化器：x/dy per-tensor 动态（Triton 融合 kernel，div+round+clamp+cast 一次读写）、w per-output-channel。`q`（round）路径用 STE。

### `torch._int_mm` 的硬件约束（调试实录，部分已用 gemmEx 绕过）

1. **行数必须是 32 的倍数**（实测 144 行 FAIL / 160 行 OK；此前以为"须 >16"是错觉）→ pad 或换 `cublasGemmEx`（transb=T 免转置）；
2. **K 必须是 8/32 倍数**（im2col 后 `K=C·9`，如 27 → pad 到 32）→ `F.pad` 注意参数从最后一维开始排布，pad 行/列与另一因子的 pad 列/行对齐；
3. **行数上限约 32K**（`49152×k` 可以、`65536×k` 挂/错）→ 首维分块 32768 后再 `cat`；
4. CUDA 上 **int64 matmul 不可用**（CPU 模拟才走 int64 累加路径）；
5. STE round 使数值 gradcheck 不适用（分段常数导数），改用逐张量相对误差断言。

### 数值验证（GPU 真 kernel，PASS）

| 检查项 | 相对误差 |
|---|---|
| GEMM 前向 | ~1.3% |
| Conv 前向（cuDNN int8） | ~1.5% |
| 权重梯度 dW（SwitchBack int8，gemmEx） | ~1.2% |
| 输入梯度 dX | 3e-4（CUDA fp32 累加顺序差，其余 0.0）；方案 C（E 量化）~1.2% |

整模型（yolov8n，COCO128）sanity PASS：45/64 层转 int8、loss 有限。

### 方案 C：全操作数 int8（E 量化）实现与闭环

`--quantize-e` 开关把方案 C 补齐：**dX 链的误差信号 E 也走 int8 GEMM**（`dyq @ wqᵀ`，dy 复用 dW 的 dyq 零额外量化、w 用 per-tensor wq 保证 scale 可提出，`F.fold`/col2im 还原），实现"前向 + dW + dX 三链全 int8"的完整训练引擎：

| 指标 | 方案 B（dX fp） | 方案 C（E 量化） |
|---|---|---|
| dx 相对误差 | 0.0（精确） | **1.17%**（dy+w 量化） |
| 整模型梯度流通 | 183/184 | 183/184（全有限） |
| 吞吐（n@320/bs16） | 73.2 ms | 78.4 ms（更慢：小 K GEMM 分块 + fold 开销） |

结论：方案 C 把工作 I/J 假量化结论在**真 int8 引擎**上闭环——E 量化引入 1.17% 的 dX 误差（即工作 I/J 掉 1.1-3.4 点的根源），且吞吐更差。**全 int8 训练链工程可行但无收益，方案 B 仍是推荐配置**。

（附：smoke 的 183/184 梯度 = box/dfl 损失在随机噪声输入下为 0 所致——TaskAlignedAssigner 分不到正样本，`box_loss`/`dfl_loss` 恒 0，cv2 分支梯度全零、dfl 梯度 None。真实数据训练时全部 184 个参数正常收到梯度，非引擎问题。）

### 训练精度验证（COCO128, 320px, bs16）与三个隐藏 bug 的修复

引擎首次真实训练暴露了三个 sanity 覆盖不到的 bug（全部已修复并有回归测试）：

1. **dW int32 累加溢出**：dW 收缩维 K=NL（160px 层 409600），`409600×127×127 = 6.6e9 > 2^31` → 梯度爆炸（grad_norm 4058，权重被破坏、mAP 崩到 0.004）。sanity 未暴露（测试规模 NL≤4096）。修复：沿 K 分块 32768 + fp32 累加（回归测试与 fp64 参考逐位一致）。
2. **Triton quant 对非连续输入错乱**：`x.reshape(-1)` 对 permute 视图返回非连续 1D 视图，kernel 裸指针按逻辑索引读错位；`empty_like` 对非连续输入保留 stride 使输出也错位。修复：输入/输出显式连续（`contiguous()` + `torch.empty(shape)`）。
3. **`_cudnn_nhwc_stride` 布局算错**：C-stride 应为 1、H-stride 应为 W×C，旧实现全错 → C=64 等形状 cuDNN 输出布局错乱（model.5 起 rel 100% 误差）。修复后单层误差回到 1.4%。

修复后完整训练对比（seed=0 确定性）：

| 训练 | 10 epochs mAP50-95 | 50 epochs mAP50-95 | 50 epochs 掉点 | 训练步 |
|---|---|---|---|---|
| 真 fp32 | 0.445 | **0.526** | 基准 | ~78 ms |
| AMP | 0.443 | —（与 fp32 持平，官方默认配置） | ~0 | ~40 ms |
| engine B（int8） | 0.361（-8.3 点） | **0.510** | **-1.7 点** | ~125 ms |
| engine C（E 量化） | 0.362（-8.3 点） | **0.503** | **-2.3 点** | ~127 ms |

结论：

- **网格适配假设成立**：10→50 epochs 掉点从 -8.3 收窄到 -1.7——训练中权重向 int8 网格靠拢（工作 G 的 QAT 机制），量化噪声被吸收；
- **真 int8 引擎 vs 假量化 QAT**（工作 F：-0.09 点）：真引擎略逊（每层 1.4% 硬量化误差逐层累积，feats 到深层 rel ~36%），但量级可接受——int8 训练引擎"精度可行、速度不划算"（0.6x 慢）是最终结论；
- **AMP 精度实测闭环**：10 epochs 与 fp32 差 0.002（噪声级），官方默认配置的"不掉点"说法实测支持。

### 涨点尝试：Distribution Adaptive INT8（论文 2102.04782）与 GVQ/EMA

参考阿里达摩院论文 *Distribution Adaptive INT8 Quantization for Training CNNs*（2102.04782）——其核心贡献是梯度**按输出通道分别量化**（GVQ，替代我们的 per-tensor 全局量化）+ **Magnitude-aware Clipping**（长尾分布的 scale 用 EMA 平滑，有正则化效应），论文在 ImageNet 分类上 INT8 训练甚至微涨（ResNet-50 +0.09%）。据此给引擎加了 `--gvq` / `--ema-scale`（含论文的 P(|g|>σ)≤0.3 分布判别）开关：

**COCO128 50 epochs（seed=0 确定性，噪声底 ±0.002）**：

| 训练 | mAP50 | mAP50-95 |
|---|---|---|
| fp32 | 0.690 | 0.526 |
| engine-B | 0.676 | 0.510 |
| GVQ（dy 通道量化） | 0.677 | 0.511 |
| GVQ + EMA（无条件/自适应） | 0.679 | 0.509 |

**结论：GVQ/EMA 在 COCO128 上无改善**（全部在噪声内）。原因：论文涨点全部在分类任务（检测任务论文自己也只是"几乎无损"），且 COCO128 的 ±0.005 噪声底大于论文的 +0.3% 增益。

**COCO2017 全量（bs32）的意外发现——GVQ 的稳定性价值**：

| COCO2017 bs32 + lr0=0.005 10 epochs | mAP50-95 |
|---|---|
| engine-B | **崩**（0.001，loss 先升后降、val 先升后崩） |
| **GVQ** | **0.140（稳定收敛）** |

GVQ 的 per-channel 梯度量化在 bs32 下显著更稳（bs32 时 engine-B 的 per-tensor dy 量化不稳定——表现为 10 epochs 内 val 单调崩；bs16 下两者均稳定）。**这是论文"梯度分布感知"论点的实证**：per-channel 量化不仅减小误差，还提供训练稳定性。

（记录：engine 对 lr 敏感——bs32 + 默认 lr0=0.01 崩得更早；bs32 的崩因未完全定位，bs16 为已验证可靠配置。COCO2017 bs16 的 GVQ vs engine-B 精度对比进行中。）

### 三轮工程迭代的吞吐轨迹（batch=16, imgsz=320, 同机对比）

| 版本 | engine ms/step | vs FP32 | 累计改动 |
|---|---|---|---|
| ① 初版 im2col + `torch._int_mm` | 74.5 | 0.34x | unfold+转置+`_int_mm`，转置搬运 22ms 是最大单项 |
| ② cuDNN INT8 fprop | 73.7 | 0.35x | fprop 换 cuDNN 图 API（单层 3-4x 快），但 dW 的 unfold+转置仍在 |
| ③ int8 unfold 新路径 | 71.0 | — | unfold 后立即 cast int8 + `permute(1,0,2).reshape` 转置（31→6.9ms/160 层） |
| ④ gemmEx 免转置 dW + Triton quant 融合 | 71.4 | **0.57x** | 自写 CUDA ext（`cublasGemmEx` transb=T）+ Triton quant kernel（单层 0.9→0.008ms） |

最终同状态实测：FP32 原生 40.5ms（该实例后续被限功率 ~115W，满血时为 26ms）| engine 71.4ms。

### 大模型验证（限功率状态下相对值）

| 模型 | imgsz/bs | fp32 | engine | 比值 |
|---|---|---|---|---|
| yolov8n | 320/16 | 40.5 | 71.4 | 0.57x |
| yolov8l | 320/16 | 111.7 | 270.8 | 0.41x |
| yolov8x | 320/16 | 175.3 | 381.0 | 0.47x |

### 大模型为什么没能翻盘：三个理论假设的破灭

"模型越大 int8 越该赢"的理论链条是：int8 tensorcore 算力 8x → GEMM 更大更算力受限 → 优势显现。实测证明三个隐含假设全部不成立：

1. **假设"所有卷积都走 int8 高效 kernel"——错，1x1 层是负资产**。yolov8 卷积里 1x1 占 ~40%（cv1/cv2 全是）。单层实测（yolov8x）：3x3 大层（320→320、640→640）int8 快 **1.9-2.3x**；1x1 层（如 1600→640）int8 慢 **0.3-0.7x**——cuDNN 的 int8 1x1 kernel 无 IMMA 优化，而 fp32 1x1 是纯 cuBLAS GEMM 极快。模型越大 1x1 越重，3x3 省的钱被 1x1 亏的钱吃掉大半。

2. **假设"训练 = 大 GEMM"——错，反传占 67% 且 int8 化无门**。yolov8x 实测 fwd 58.7ms / bwd 116.6ms。反传两条链：dX 在 sm86 上 cuDNN 的 INT8 dgrad/wgrad 直接 NOT_SUPPORTED（实测），只能 fp32 conv_transpose——它本身就是高效隐式 GEMM，int8 零收益；dW 的 SwitchBack 数学可行，但 unfold 展开+内存搬运按 FLOPs 同比例增长。

3. **假设"kernel 时间 = 算力时间"——错，带宽与逐元素操作占大头**。3x3 大层也只有 2.3x（非 8x）：40px 分辨率下 GEMM 不够大，kernel 带宽受限而非算力受限。BN/SiLU 逐元素（~12ms）、quant 扫描（~15ms）、python 调度（~50ms）全都不随 int8 加速，纯开销。

yolov8x@320 的 381ms 拆解：fwd ≈ quant 15 + cudnn fprop ~40（3x3 快/1x1 亏，净亏）；bwd ≈ dW 展开+GEMM ~150 + dX fp ~40 + quant 10 + 调度 ~50。**能在 int8 上赢钱的只有 fprop 的 3x3 大层（~30ms 量级），亏损项是它的 5-10 倍**。imgsz 320→640 时比值 0.47→0.49 略升（GEMM 更算力受限），方向对但远远不够。

要让 int8 训练真正赢，缺的不是模型大小，而是 cuDNN 在 sm86 不给的三样东西：1x1 的 int8 kernel、dgrad/wgrad 的 int8 kernel、整图执行（省调度）。

### 最终瓶颈：Python 调度（语言层面的结构性天花板）

`torch.profiler` 实测 engine 一步：**CUDA kernel 40ms + CPU 调度 35ms**（≈ 88% 并行度缺失，二者几乎串行）：

- CUDA 侧（40ms）：dW 的 `int_mm` 6.9ms + unfold 3.6ms + BN/SiLU 逐元素 ~12ms + quant 扫描 1.1ms + cuDNN fprop ~2ms + dX ~3ms……
- CPU 侧（35ms）：45 层 × 每层 ~15 次 kernel 启动（量化 1 + execute 1 + sdy 2 + 量化 1 + unfold 2 + cast 1 + 转置 1 + gemmEx 1 + convT 2 + scale 2……）的 Python dispatch 开销。kernel 启动 ~40-80μs/次 × ~500 次 ≈ 35ms

**结论（诚实负结果）**：

1. int8 单 kernel 收益是真实的（cuDNN fprop 大层 2-4x、gemmEx 免转置、quant 融合 100 倍），三轮工程把 engine/fp32 从 0.34x 改善到 0.57x；
2. **但 Python 调度 35ms 是语言层面的墙**：PyTorch 手工引擎每层 ~15 次 kernel 启动，启动开销总量超过所有 int8 kernel 收益之和。想跨过它只剩"整模型编译成单个 cuDNN/CUDA 图"（一次 execute 全网络）——那相当于重写一个训练版 TensorRT，超出本工作射程；
3. 与工作 K（推理部署）同构：int8 加速必须靠"整图融合"兑现，裸 kernel 替换在 Python 层拿不到。**3080 Ti + yolov8 场景下 INT8 训练引擎工程上不划算**，这是本项目的最终判断。

### FP16 AMP 对照：训练加速的现实路径（一行代码 vs 三轮工程）

对"训练加速"本身而言，成熟工具链给出了更优答案——PyTorch AMP（fp16 混合精度，`torch.autocast` 一行）。同机同状态实测（限功率）：

| 模型 | imgsz/bs | fp32 | AMP | INT8 引擎 | AMP 加速 |
|---|---|---|---|---|---|
| yolov8n | 320/16 | 33.1 | 38.4 | 71.4 | **0.86x（反而慢）** |
| yolov8l | 320/16 | 111.9 | 64.5 | 214.9 | **1.73x** |
| yolov8x | 320/16 | 174.8 | 108.3 | 294.5 | **1.61x** |
| yolov8x | 640/8 | 359.6 | 187.5 | 790.8 | **1.92x** |

三个观察：

1. **AMP 完胜 INT8 引擎**：yolov8x@320 上 AMP 108ms vs int8 引擎 294ms，快 2.7 倍——`torch.autocast` 一行打败了工作 L 全部三轮工程。int8 的 8x 算力优势被调度/量化/展开吃掉后，不如 fp16 的 2x 内存优势兑现得干净；
2. **yolov8n 上 AMP 也慢（0.86x）**——与 int8 引擎同构的病：小模型 kernel 启动/调度开销主导，类型转换开销 > fp16 kernel 收益。带宽/调度受限场景下"精度减半"的收益兑现不了，模型/分辨率足够大（算力受限）后才显现（640px 达 1.92x）；
3. **AMP 基本不掉点**：`amp=True` 是 ultralytics 官方默认训练配置（官方 COCO 权重全部是 AMP 产物，与 fp32 差异 <0.1 点噪声级）。机制上 fp16 是"软降级"——权重保持 fp32 master copy、梯度 fp32 累加 + GradScaler、BN 走 fp32，每步误差 ~1e-3 不累积；对比 int8 的"硬降级"（工作 I/J 的 E 量化掉 1.1→3.4 点、STE 近似、逐层累积）。

**项目最终闭环**：训练加速用 AMP（大模型 1.6-1.9x、零工程、零掉点）；int8 的正确位置在推理部署（工作 K，TRT INT8 收益有限但真实）。INT8 训练引擎在本场景的结论是硬负结果——但它的全部中间证据（调度墙、1x1 无 kernel、dgrad/wgrad 无 INT8、int32 落地 4x 流量、CUDA 13.1 无 EPILOGUE_SCALE）对任何想再做 int8 训练的人都是完整的地图。

复现（GPU 服务器，需 CUDA torch + ninja + nvcc）：

```bash
python int8_engine.py --sanity          # 数值验证（CPU 模拟 / cuda_intmm 双后端）
python bench_final.py                   # fp32 / im2col / engine 三模式吞吐对比
python int8_engine_train.py sanity      # 整模型 sanity
python int8_engine_train.py train       # ultralytics 流水线接入（--epochs/--data/--imgsz）
```

### 最终定位：真 int8 引擎是伪量化 QAT 的对照验证实验

项目至此闭环。对"int8 训练引擎是否必要"的最终回答：

1. **伪量化就是"量化操作数"的标准实现**。"伪"指硬件执行（乘法用 fp32 模拟 round 后的 int8 值），操作数本身是**真 int8 值**（`W_q`/`A_q`/`E_q` ∈ [-128,127] 网格）——工作 G 的四层硬证据（45 个量化卷积、每 batch 90 次量化调用、消融开关、网格误差 0.39%）证明这一点。Google/NVIDIA 的 QAT 都是伪量化。

2. **真 int8 引擎（工作 L）验证了伪量化的合理性**：
   - 数值一致性：真 int8 误差 1.2% vs 伪量化 1.3%（差 0.1%）——伪量化的数值模拟足够精确；
   - 硬件不可行：真 int8 在消费 GPU 上训练 0.6x 慢（调度墙 + 带宽浪费 + 工具链缺功能）——训练侧无 int8 硬件收益，**生产路径采用伪量化 QAT（几乎无损）+ TensorRT INT8 推理部署（QAT 权重只掉 1.3%）**。

3. **本项目的最终形态**：量化操作数（伪量化 QAT，A-K）→ 验证与对照（真 int8 引擎，L）→ 落地（TensorRT INT8 部署，K）。int8 的正确位置在推理侧；训练侧的正确工具是 QAT 让权重对 int8 免疫。

### kernel 级解耦实验：int8 快在 GEMM，死在量化和调度两道税

为回答"int8 训练到底慢在哪"，把开销逐层剥离（3080 Ti, bs16, imgsz320）：

**纯 GEMM（预量化输入，不含量化扫描）**：

| 形状 | fp32 matmul | int8 GEMM | int8 加速 |
|---|---|---|---|
| 160px | 2.185 ms | 1.063 ms | **2.06x** |
| 40px | 1.289 | 0.515 | **2.50x** |
| 20px | 1.602 | 0.494 | **3.24x** |

**int8 kernel 层面确实快 2-3 倍**（tensorcore），但完整引擎反而慢：

| 引擎 | 训练 ms/step | 推理 ms/step | 训练峰值显存 |
|---|---|---|---|
| PyTorch 原生 FP32（cuDNN/cuBLAS，英伟达优化） | 40.1 | 6.7 | 0.81 GB |
| FP32 手搓引擎（im2col+matmul，同架构） | 56.6 | 20.9 | 1.38 GB |
| **int8 引擎**（真 int8 kernel） | 80.5 | 30.8 | **0.74 GB** |
| QAT W+A+E（伪量化） | 105.3 | 41.7 | — |

**三层结论**：

1. **量化税**：单层 absmax 扫描 + round/clamp/cast 耗时 0.544ms（160px），把 int8 的 2.06x GEMM 优势吃掉 1/3（含量化后只快 1.36x）；叠加 dW 转置、dX 展开、python 调度，完整引擎反慢 1.42x；
2. **显存是 int8 唯一明确优势**：同架构下小 46%（int8 激活 4 倍小），相对原生 FP32 小 9%——在大 batch/高分辨率下会放大；
3. **最终判断：YOLO 的训练没有必要真 int8 量化**——FP32 更快（用英伟达工程师优化过的 cuDNN/cuBLAS 训练引擎）、精度更好、无需自研 kernel。因此本项目通过**增加量化/反量化节点进行伪量化**（QAT）：操作数确为 int8 值（工作 G 硬证据），训练几乎无损，部署走 TRT INT8 兑现硬件加速。

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