# CUB-200 鸟类图像分类

《模式识别与机器学习》课程大作业，包含给定特征上的传统分类和原始 RGB 图像上的深度学习分类。

## 任务与进度

| 任务 | 数据范围 | 输入 | 进度 |
| --- | --- | --- | --- |
| 传统分类 | 固定的 10 个类别 | 教师提供的特征向量 | 已实现数据检查、固定划分、Softmax 训练与评估；SVM 待实现 |
| 深度学习分类 | 完整 200 个类别 | 原始 RGB 图像 | 待实现 |

主要指标为 Top-1 准确率。两项任务的类别数和输入表示不同，结果分别报告。[项目方案](docs/project-plan.md)记录整体设计与实施阶段。

[首轮实验记录](docs/experiments/softmax-baseline.md)：10 类 Softmax 在 78 张内部验证样本上正确分类 78 张；最终测试集尚未评估。该结果来自单个固定划分，不代表完整 200 类性能。

## 环境

Python 3.10 或更新版本。传统分类使用 CPU，PyTorch 仅负责读取课程提供的 `.pt` 特征；Softmax 的损失、梯度、优化和预测均使用 NumPy 实现。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements-lock.txt` 保存首轮 macOS arm64、Python 3.13 实验环境的精确版本；其他平台按 `requirements.txt` 安装兼容版本。

以下命令从仓库根目录执行。数据和输出路径均可通过命令行指定。

## 数据准备

从[课程数据链接](https://cloud.tsinghua.edu.cn/f/72aab178f61948c095dd/)下载 `data.zip`，解压后将数据放置为：

```text
data/
  train/
    <class>/
      <sample>.jpg
      <sample>.pt
  val/
    <class>/
      <sample>.jpg
      <sample>.pt
```

使用教师提供的数据版本。`val/` 是最终测试集；调参和检查点选择使用从 `train/` 按类别划出的内部验证集。

首次生成数据检查报告与固定划分：

```bash
python -m src.prepare_data --data-dir data
```

检查覆盖图片解码、图像/特征配对、特征维数与有限值、类别一致性，以及完全相同图片的跨划分泄漏。同类训练重复图片保持在同一子集中；原始训练与测试间的重复图片或冲突标签会阻止生成清单。检查阶段只核验测试数据的完整性，不训练或计算测试性能。

默认从 200 类中按种子 42 固定抽取 10 类，完整 200 类映射也保存在清单中。每个类别约 15% 的原始训练样本划入内部验证，至少保留一个拟合样本与一个验证样本。哈希和相对路径将实际文件与划分绑定。

输出为 `outputs/audit.json` 和 `splits/cub200-seed42.json`，已有文件不会被覆盖。仓库已有清单时可直接使用，不必重新生成；如需核验完整数据，可指定临时输出并比较清单哈希：

```bash
python -m src.prepare_data --data-dir data \
  --output outputs/recheck/manifest.json \
  --audit-output outputs/recheck/audit.json
```

默认 `.pt` 为一维浮点 Tensor，也接受仅含单例额外维度的向量。字典格式通过 `--feature-key` 显式指定字段。文件始终以 `weights_only=True`、CPU 方式加载。未知格式会报告错误，不自动展开空间特征图。

## 训练 Softmax

```bash
python -m src.train_traditional --data-dir data \
  --manifest splits/cub200-seed42.json \
  --config configs/softmax.json \
  --output-dir outputs/softmax-seed42
```

目标函数为平均交叉熵加 `0.5 × l2 × ||W||²`，偏置不作正则化。标准化参数只由拟合子集估计，常量维度的缩放系数设为 1。

默认训练 100 轮，batch size 64，学习率 0.01，比较 `l2 ∈ {0, 1e-4, 1e-3, 1e-2}`。各配置使用相同初始化和小批量顺序。优先选择内部验证准确率最高的检查点，相同时选择验证交叉熵更低的检查点；完全相同时保留先出现者。选择过程不读取测试特征。

日志中的训练和验证损失均为不含正则化项的平均交叉熵。

每次运行使用新的输出目录，包含：

| 文件 | 内容 |
| --- | --- |
| `config.json`、`manifest.json`、`environment.json` | 参数、划分快照、版本与运行环境 |
| `history.csv` | 各配置逐轮训练和内部验证损失、准确率 |
| `candidate-*.npz` | 各正则化配置的最佳模型 |
| `model.npz` | 按内部验证选出的最终模型 |
| `summary.json` | 选择结果、样本数、耗时与内部验证混淆矩阵 |
| `internal_val_predictions.csv` | 内部验证逐图预测 |

模型包含标准化参数、权重、偏置、类别顺序和划分哈希。模型读取不使用 pickle。此阶段支持保存模型后独立评估，尚不支持中途恢复训练。

## 独立评估

重新加载保存模型，复核内部验证结果：

```bash
python -m src.evaluate --data-dir data \
  --model outputs/softmax-seed42/model.npz \
  --split internal_val --output-dir outputs/softmax-validation
```

模型与清单的哈希、类别顺序必须一致；加载的特征与清单中的文件哈希不一致时会报错。

最终方案冻结后，单独运行测试评价：

```bash
python -m src.evaluate --data-dir data \
  --model outputs/softmax-seed42/model.npz \
  --split test --output-dir outputs/softmax-test
```

测试结果不用于继续调参。输出包含准确率、正确数/总数、类别顺序、混淆矩阵、模型哈希及逐图预测。JSON 中准确率取值为 0–1，命令行显示百分比。当前模型输入为提供特征，不提供从任意 RGB 图像提取教师特征的功能。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试包含数值梯度、极端 logits、常量特征、损坏与缺失文件、重复图片隔离、文件变动检测、划分复现、训练及模型恢复。端到端测试在生成划分后移除测试数据，验证训练和内部验证流程仍可独立完成。

## 目录

```text
configs/                  训练配置
splits/                   固定划分和类别映射
src/
  data.py                 数据检查、划分及特征加载
  prepare_data.py         数据准备命令
  traditional/softmax.py  标准化、Softmax 损失及模型
  train_traditional.py    训练和内部验证选择
  evaluate.py             保存模型的独立评价
  utils.py                序列化、指标和环境记录
tests/                   算法与流程测试
docs/                    项目方案和实验记录
```

原始数据、环境、缓存和实验产物由 `.gitignore` 排除。代码、配置、固定划分与精简实验记录纳入版本控制。
