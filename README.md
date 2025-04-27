# PKU_Deeplearning_Lab3

# 任务说明：

1. Street View House Number Recognition

* 数据集：http://ufldl.stanford.edu/housenumbers/
* SVHN（Street View House Number）Dateset 来源于谷歌街景门牌号码，本次作业的目标数据集是其中的 Format 1 (Full Numbers: train.tar.gz, test.tar.gz , extra.tar.gz). 其中，train.tar.gz 为训练数据集，test.tar.gz为测试数据集。注：extra.tar.gz是附加数据集，建议不使用。
* 在train.tar.gz与test.tar.gz中，分别包含：（1）一些.png格式的图片，每张图片包含一个门牌号；（2）一个digitStruct.mat文件，包含每张图片所对应的门牌号，以及每个门牌号数字的位置信息；（3）一个see_bboxes.m文件，用于辅助Matlab环境下的处理，请忽略之。
* 作业要求：

> 1. 设计一个网络，用train.tar.gz中的数据进行训练，并用test.tar.gz中的数据进行测试；
> 2. 在测试的过程中，不允许使用test.tar.gz/digitStruct.mat文件中的位置信息作为输入，即必须在“忽略测试数据集中给出的位置信息”的前提下，识别出test.tar.gz中每张图片中的门牌号；
> 3. 撰写一个PPT，汇报如下信息：

>> （1）所设计的网络的结构和超参数信息；
>>
>
>> （2）网络的训练方法和优化方法；
>>
>
>> （3）体现训练过程的“训练曲线”；
>>
>
>> （4）识别准确率；
>>

# 项目文件结构

```
📦 PKU_DeepLearning_Lab3
 ┣ 📂 yolo11                       # yolo11的相关代码
 ┣ 📂 CRNN                       # 用CRNN的相关代码
 ┣ 📂 CNN                       # 用CNN的相关代码

```

# CRNN方法

## 模型架构

本模型是一个面向**不定长门牌号数字串识别**的深度学习系统，基于**Attention-CRNN结构**设计，整体由**卷积特征提取、注意力机制、序列建模、CTC Loss训练**四大模块组成。具体流程如下：

------

### 1. 输入处理（Input Preprocessing）

- 输入为尺寸为 **64×64×1** 的灰度图像。
- 初始加入**Gaussian Noise（标准差0.01）**，用于模拟拍摄噪声，提升模型对输入扰动的鲁棒性。

------

### 2. 卷积特征提取模块（CNN Backbone）

- 使用四层卷积块进行特征提取，每层包含：
  - 两次 **Conv2D（无偏置项） → Batch Normalization → ReLU 激活**；
  - 后接 **MaxPooling** 降低空间尺寸；
  - 再接 **Dropout** 抑制过拟合（Dropout率随层数逐步增大，从0.2到0.35）。
- 卷积通道数依次加深：32 → 64 → 128 → 256。

**作用**：逐步提取图像的低层次到高层次视觉特征（如局部边缘、局部模式、复杂图形组合）。

------

### 3. 注意力机制模块（Attention Module）

- 在卷积提取后的特征图上插入自定义**Attention机制**。
- 该模块可动态调整特征图中各区域的权重，引导模型更关注于图像中关键的门牌数字区域，减少背景噪声干扰。

**作用**：提升特征表示的判别性和鲁棒性，尤其对复杂背景图像有效。

------

### 4. 特征序列化（Feature Reshaping）

- 将卷积后的二维特征图，**Reshape**成一维时序特征序列（时间步 × 特征维度），
- 为后续序列建模（RNN）模块做准备。

**作用**：将空间特征转化为序列特征，便于捕捉序列内部的时序关系。

------

### 5. 序列建模模块（BiLSTM堆叠）

- 堆叠**四层双向LSTM（Bidirectional LSTM）**，每层256单元。
- 每层LSTM后加入**Layer Normalization**，提升深层递归网络训练稳定性。
- 所有LSTM使用：
  - He Normal初始化（kernel_initializer='he_normal'）
  - Orthogonal递归权重初始化（recurrent_initializer）
  - L2正则化（1e-4），并启用梯度裁剪（clipnorm=1.0）。

**作用**：从前向和后向两个方向同时建模门牌号序列中字符的上下文依赖关系。

------

### 6. 输出层（Dense + Softmax）

- 对序列中每一个时间步，输出一个**nb_classes维**（比如11类，包括0-9数字和blank）的概率分布。

------

### 7. 损失函数（CTC Loss with Label Smoothing）

- 使用**CTC（Connectionist Temporal Classification）Loss**，允许输出长度与真实标签长度不一致，自动学习对齐关系；
- 同时引入**Label Smoothing**技术，缓解模型过度自信，提高泛化能力。

![1745760716246](assets/1745760716246.png)

## 训练超参数表

| 超参数类别            | 参数名                                | 设置值     | 说明                             |
| --------------------- | ------------------------------------- | ---------- | -------------------------------- |
| **学习率调度**        | 初始学习率 (`initial_learning_rate`)  | 0.001      | 训练初期的学习率起点             |
|                       | Warmup轮数 (`warmup_epochs`)          | 5          | 前5个epoch线性升高学习率         |
|                       | Cosine衰减周期 (`decay_epochs`)       | 20         | 余弦退火下降的总epoch数          |
| **Early Stopping**    | 监控指标 (`monitor`)                  | `val_loss` | 以验证集损失为监控对象           |
|                       | 耐心等待轮数 (`patience`)             | 3          | 验证集性能无改进时提前停止训练   |
|                       | 最小改善量 (`min_delta`)              | 1e-4       | 损失必须至少改善1e-4才能重置耐心 |
|                       | 恢复最佳权重 (`restore_best_weights`) | True       | 停止时回到验证集表现最好的权重   |
| **ReduceLROnPlateau** | 缩放因子 (`factor`)                   | 0.5        | 学习率降低为原来的一半           |
|                       | 耐心等待轮数 (`patience`)             | 2          | 验证集性能无改进时降低学习率     |
|                       | 最小改善量 (`min_delta`)              | 1e-4       | 损失改善低于1e-4认为无效         |
|                       | 最低学习率 (`min_lr`)                 | 1e-6       | 学习率不会降低到低于此值         |
| **训练过程**          | 总训练轮数 (`epochs`)                 | 40         | 最大训练轮数                     |

# Yolo方法

## 使用框架

https://docs.ultralytics.com/usage/python/
