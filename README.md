# Image-Classification-with-Pretrained-Model

## 1. 项目概述

本项目旨在利用 PyTorch 框架，基于预训练的 ResNet-18 模型对 Caltech101 数据集进行图像分类。
代码实现了数据集的自动划分、加载、预处理、模型的构建、分阶段训练（包括冻结和解冻层进行微调）、验证和测试。同时，项目还包含了超参数的网格搜索以及预训练模型与非预训练模型的对比实验。

## 2. 环境准备

本项目需要 PyTorch 及其相关的库。

1.  **创建虚拟环境：**
    ```bash
    conda create -n pytorch_env python=3.8 -y
    conda activate pytorch_env
    ```

2.  **安装 PyTorch 和 TorchVision：**
    请根据您的操作系统和 CUDA 版本访问 PyTorch 官方网站获取正确的安装命令。

3.  **安装其他依赖：**
    ```bash
    pip install matplotlib tensorboard scikit-learn
    ```

## 3. 数据集准备

数据集 Caltech101 是一个包含 101 个类别的图像数据集，每个类包含数百张图像，用于分类任务。

1.  **下载 Caltech101 数据集：**
    从 [Caltech 101 官方网站]("https://data.caltech.edu/records/mzrjq-6wc02") 下载数据集。

2.  **组织数据集：**
    将下载的数据集解压。确保数据集的根目录包含各个类别的子文件夹（例如 `101_ObjectCategories`）。

3.  **设置数据路径：**
    在脚本中的主程序中，需要指定`work_dir`变量为您的工作目录，并修改 `data_dir` 变量为您数据集的实际路径：
    ```
    data_dir = '替换为数据集所在目录\\101_ObjectCategories' 修改为您的数据集路径
    ```

    脚本会自动将数据集划分为训练集、验证集和测试集，并保存在 `work_dir` 指定的目录下。

## 4. 代码结构

代码包含以下主要函数：

* `split_dataset`: 将原始数据集按比例划分为训练集、验证集和测试集。
* `load_dataset`: 加载划分好的数据集，并应用数据增强和预处理。
* `build_model`: 构建 ResNet-18 模型，支持加载预训练权重。
* `train_model`: 执行模型的训练过程，记录训练损失、验证损失和验证准确率。
* `validate_model`: 在验证集上评估模型性能。
* `test_model`: 在测试集上评估模型最终性能。
* `train_test_pretrained_model`: 针对预训练模型，实现分阶段（冻结/解冻层）的训练和测试流程。
* `train_test_no_pretrained_model`: 训练和测试不使用预训练权重的模型。

主程序 (`if __name__ == '__main__':`) 负责设置工作目录、加载数据集、执行超参数网格搜索和预训练/非预训练模型的对比实验。

## 5. 超参数网格搜索

脚本中对以下超参数进行了网格搜索：

*   `batch_size`: [16, 32, 64]
*   `learning_rate`: [1e-2, 1e-3, 1e-4]
*   `weight_decay`: [1e-3, 1e-4, 1e-5]

每个组合的训练过程和结果都会记录在独立的 TensorBoard 日志目录中，方便对比分析。

## 6. 预训练与非预训练模型对比

脚本最后会对比使用预训练 ResNet-18 和从头开始训练的 ResNet-18 在相同设置下的性能。结果将通过 TensorBoard 记录在 `pretrained_resnet18` 和 `no_pretrained_resnet18` 目录下。

## 7. 如何运行

1.  激活您的虚拟环境：
    ```bash
    conda activate pytorch_env
    ```

2.  运行脚本：
    ```bash
    python Caltech_101.py
    ```

脚本将依次执行超参数网格搜索和预训练和非预训练模型的对比实验，可根据您的要求注释掉相关代码，训练过程中的日志和结果将输出到终端，并通过 TensorBoard 记录。

## 8. 实验结果与可视化

实验结果和 TensorBoard 日志将保存在 `work_dir` 指定的目录下。

*   **TensorBoard 可视化：**
    在终端中，导航到 `work_dir` 目录，并运行 TensorBoard 命令：
    ```bash
    tensorboard --logdir=work_dir # 替换为您的工作目录
    ```
    打开浏览器访问 TensorBoard 提供的地址（通常是 `http://localhost:6006`）即可查看训练损失、验证损失和验证准确率等曲线。

*   **最佳模型：**
    训练后的最佳模型权重将保存为 `best_model.pth` 在相应的工作目录下。

*   **测试准确率：**
    最终的测试准确率将打印到终端。

库中的`log`文件夹为实验过程中tensorboard记录的训练日志，可供参考。
