# 200 类 CNN 训练与复现

## 输入与结构

模型从原始图片随机初始化训练，不加载提供的 `.pt` 特征，不使用预训练权重。类别映射采用固定清单中的全部 200 类，与传统分类的 10 类名单分开。

图像首先转换为 RGB，按比例缩放到 `160×160` 画布内，再居中填充灰色像素 `(128,128,128)`。像素变换为 `pixel / 127.5 - 1`。该方式保留完整图像与宽高比，归一化采用固定常数。D0 基线的训练、验证和推理均使用这一确定性预处理，无随机增强。

| 阶段 | 操作 | 默认输出形状，不含批量维 |
| --- | --- | --- |
| 输入 | RGB 图像 | 3×160×160 |
| 卷积阶段 1 | 3×3 卷积，stride 2；3×3 卷积，stride 1 | 32×80×80 |
| 卷积阶段 2 | 同上 | 64×40×40 |
| 卷积阶段 3 | 同上 | 128×20×20 |
| 卷积阶段 4 | 同上 | 256×10×10 |
| 池化 | 全局平均池化 | 256 |
| 分类头 | 线性层 | 200 logits |

每个卷积后包含 BatchNorm 和 ReLU，padding 为 1，卷积不使用偏置。卷积参数采用 Kaiming 随机初始化，分类头采用标准差 0.01 的正态初始化。默认参数量为 1,224,616。

## 训练

```bash
python -m pip install -r requirements.txt
python -m src.train_cnn --data-dir data \
  --config configs/cnn-baseline.json \
  --output-dir outputs/cnn-baseline --device auto
```

`auto` 依次选择 CUDA、MPS、CPU；也可显式指定设备。训练始终读取固定清单的全部拟合子集和内部验证子集，不提供缩减正式训练样本的隐藏开关。`data/val/` 不参与训练和模型选择。

默认配置为 batch size 64、SGD、初始学习率 0.05、momentum 0.9、weight decay 1e-4、80 轮余弦衰减。交叉熵直接接收 logits。训练指标是在每轮参数持续更新过程中累计的值；验证指标来自该轮结束后固定模型的评估。损失按样本数加权平均，不让最后一个较小批次获得额外权重。

每轮末尾执行内部验证。先比较准确率，平局时比较交叉熵，完全相同时保留较早检查点。验证时关闭梯度且切换为 `eval()`，不更新 BatchNorm 统计量。训练记录不包含最终测试成绩。

## 分段运行与续训

在不改变总训练计划的前提下运行前一轮，以检查设备和耗时：

```bash
python -m src.train_cnn --data-dir data \
  --output-dir outputs/cnn-first-epoch --device auto --stop-after-epoch 1
```

从该轮结束处继续训练到配置的总轮数：

```bash
python -m src.train_cnn --data-dir data \
  --output-dir outputs/cnn-resumed --device auto \
  --resume outputs/cnn-first-epoch/last.pt
```

`--stop-after-epoch` 为绝对轮次，不改变学习率调度器的总长度。续训需要相同配置、数据清单、类别映射和设备类型；不能将推理检查点当作续训检查点。每次使用新的输出目录，以保留历史运行记录。续训后的历史文件包含原有轮次和新增轮次。

`last.pt` 保存当前模型、优化器、调度器、历史最佳模型、已完成轮次、Python/NumPy/PyTorch 随机状态及训练/验证 DataLoader 的生成器状态。GPU 运行还保存相应设备的随机状态。按轮写入临时文件后替换目标文件；中途停止时，从最后一个完整轮次恢复，未完成轮次重新执行。

CPU 与 CUDA 启用确定性算子，CUDA 同时关闭 cuDNN benchmark 并设置 cuBLAS 工作区。MPS 不承诺逐位一致。跨 PyTorch 版本、不同硬件及不同设备类型的数值结果也可能不同；单元测试核验的是同一 CPU 环境下连续训练与分段恢复的参数逐位一致性。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `config.json`、`manifest.json` | 训练参数与完整划分快照 |
| `environment.json` | 环境、设备、源代码哈希、恢复来源和本次执行轮次范围 |
| `history.csv` | 每轮学习率、训练/验证损失和准确率、分段耗时 |
| `summary.json` | 已完成轮次、是否完成训练计划、历史最佳内部验证结果 |
| `last.pt` | 最后完成轮次的完整续训状态 |
| `model.pt` | 历史最佳内部验证模型，供评估和单图预测 |

`model.pt` 和 `last.pt` 都可用于推理，分别对应最佳轮次和最后轮次；评估输出会记录检查点类型与轮次。推理文件没有优化器状态，不能用于续训。所有检查点以 `weights_only=True` 读取。

## 独立评估与单图预测

```bash
python -m src.evaluate_cnn --data-dir data \
  --checkpoint outputs/cnn-baseline/model.pt \
  --split internal_val --output-dir outputs/cnn-validation --device auto
```

评估核对检查点和清单的类别顺序与哈希，图像加载核对文件哈希。结果包含逐图预测、准确率、交叉熵、200×200 混淆矩阵及检查点哈希。最终方案冻结后，使用独立输出目录将 `--split` 改为 `test`；测试结果不用于调参。

```bash
python -m src.predict --image /path/to/bird.jpg \
  --checkpoint outputs/cnn-baseline/model.pt --top-k 5 --device auto
```

单图预测不需要训练数据目录；输出类别、标签和 Softmax 概率。这些概率未经过置信度校准。

## 绘制训练曲线

```bash
python -m pip install -r requirements-plots.txt
python -m src.plot_cnn --history outputs/cnn-baseline/history.csv \
  --output outputs/cnn-baseline/curves.png
```

该命令仅读取已有训练历史，不重新训练或改变模型。

## 验证范围

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

图像测试覆盖灰度转 RGB、保持宽高比、确定性预处理、图片变动检测，以及只读取图片而不读取 `.pt` 特征。训练测试覆盖小样本过拟合、全部类别映射、连续与恢复训练一致性、最佳模型保留、评估时 BatchNorm 不变、独立评价及单图预测。测试夹具删除最终测试集后仍能完成训练与内部验证。
