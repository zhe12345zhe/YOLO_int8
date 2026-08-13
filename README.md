# YOLO 网络梯度生成与反向传播中的操作数量化

**任务**：把 YOLO 网络梯度生成与反向传播过程中参与乘法的操作数（权重 W、激活 A、误差梯度 E）全部量化到 int8，在 COCO2017 全量数据上验证。

## 核心思想：量化"乘法操作数"，不量化"结果"

反向传播只有三条乘法链，操作数就是全部量化对象：

| 梯度生成 | 公式 | 操作数 | 结果 |
|---|---|---|---|
| 权重梯度 | `∂L/∂W = A_q ⊗ E_q` | int8 | fp32 梯度 |
| 输入梯度 | `∂L/∂X = W_qᵀ ⊗ E_q` | int8 | fp32 梯度 |
| 误差信号 | `E_q = q(E)` 逐层传播 | int8 | fp32 累加 |

Loss、BN、优化器更新、梯度存储保持 fp32——防误差累积。

## 实现：W/A 改 forward，E 改 backward

- **W**：`WeightQuant`，per-output-channel，scale=max\|W\|/127
- **A**：`ActQuant`，per-tensor 动态，scale=max\|x\|/127
- **E**：`register_full_backward_hook`，把每层往上游传的输入梯度先量化再放行（`qat_patch.py:make_e_hook`）
- 三者共用 STE 假量化：前向真量化（round/clamp 到 [-128,127]），反向直通

## 完成度验证

- 结构：45 个 QConv2d = 45 W + 45 A + 45 E 量化点
- 数学：`w.grad == unfold(A_q) × E` 误差 1.82e-05，确实用的是 int8 操作数
- 行为：每 batch 90 次量化调用、量化开关消融结果不同（量化真实生效）

## 关键结论（COCO2017 全量，15 epochs）

| 方案 | mAP50-95 | 训练速度 |
|---|---|---|
| FP32 | 0.2450 | 12.4 it/s |
| QAT W+A | 0.2394（-0.6 点，几乎无损） | 7.4 it/s |
| QAT W+A+E | 0.2110（-3.4 点，有损） | 6.8 it/s（慢 88%） |

真 int8 训练引擎（cuDNN INT8 fprop + SwitchBack dW）：数值正确（误差 ~1.2%）但 0.6x 慢、掉 1.7-2.3 点，不如英伟达官方 FP32 引擎。

**任务结论**：操作数量化可行且可验证（QAT W+A 几乎无损）；但 E 量化有损（-3.4 点）、训练显著变慢，生产路径应取伪量化 QAT + TensorRT INT8 推理部署。

## 无人机场景验证（VisDrone 微调，15 epochs）

嵌入式常用场景——无人机航拍（俯视、小目标、密集遮挡）：

| 方案 | mAP50-95 | vs FP32 | 耗时 |
|---|---|---|---|
| FP32 | 0.1414 | — | ~10.8 min |
| QAT W+A | 0.1430 | +0.16 点（几乎无损） | ~18.4 min（1.70x） |
| QAT W+A+E | 0.1237 | -1.77 点（有损） | ~20.8 min（1.93x） |

三条核心结论在第三个数据集上全部复现：W+A 几乎无损、E 量化有损（-1.1/-3.4/-1.8 点跨 COCO128/COCO2017/VisDrone 一致）、WAE 训练慢 ~2 倍。

## 快速复现

```bash
cd phase1_pytorch && python train_compare.py   # 微型 YOLO: 方案对比+操作数验证
cd phase2_ultralytics
python qat_run.py --data coco128 --epochs 50 --imgsz 320 --batch 16   # W+A
python qat_run_e.py --epochs 50 --imgsz 320 --batch 16                # W+A+E
python qat_probe_train.py --epochs 3 --imgsz 192 && python qat_prove.py  # 四层证据
python visdrone_run.py --epochs 15 --imgsz 640 --batch 16               # VisDrone 微调（GPU）
```
