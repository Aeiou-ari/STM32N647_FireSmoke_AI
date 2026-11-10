# STM32N647 边缘 AI 火焰 / 烟雾实时识别系统

基于 STM32N647（Cortex-M55 + Neural-ART NPU）的端侧实时图像分类项目：摄像头实时采集画面，板载 NPU 运行自训练卷积神经网络，将场景实时分为 **火焰 FIRE / 正常 NORMAL / 烟雾 SMOKE** 三类，并在 LCD 上输出各类别置信度。

本仓库包含从 **数据集处理 → TensorFlow 训练 → INT8 量化 → ST Edge AI 编译 → CubeIDE 构建 → 烧录部署** 的完整链路。

## 性能指标

| 指标 | 数值 |
| --- | --- |
| 网络结构 | FireSmokeCNN V3.2（Conv2D + BatchNorm + ReLU + MaxPool + GAP + Dense） |
| 输入 / 输出 | 160×160×3 UINT8 / 1×3 float32 |
| 验证集准确率 | 91.48%（早停于第 26 轮） |
| FP32 测试准确率 | 87.67%（300 张独立测试图） |
| INT8 测试准确率 | **88.00%**（量化后几乎无损失） |
| INT8 模型体积 | 201 KB |

## 目录结构

    STM32N647_FireSmoke_AI/
    ├─ training/                  # 训练脚本（按 01→07 顺序执行）
    │  ├─ 01_check_dataset.py     # 数据集完整性校验
    │  ├─ 02_clean_dataset.py      # 清洗/转 RGB/缩放
    │  ├─ 03_prepare_dataset.py   # 划分 train/val/test
    │  ├─ 04_train_model_v32.py    # TensorFlow 训练 CNN
    │  ├─ 05_evaluate_fp32_v32.py # FP32 模型测试集评估
    │  ├─ 06_quantize_int8_v32.py # INT8 全整数量化
    │  ├─ 07_evaluate_int8_v32.py   # INT8 模型测试集评估
    │  └─ splits/                 # 数据集划分 CSV
    ├─ models/
    │  ├─ int8/                   # 部署版 INT8 TFLite（烧板子用这颗）
    │  └─ fp32/                   # 训练得到的 FP32 模型
    ├─ deployment/
    │  ├─ Model/                  # ST Edge AI 编译工程配置与生成代码
    │  └─ Binary/                 # 烧录三件套 fsbl / appli / network-data
    ├─ reports/                   # 评估报告、训练曲线、混淆矩阵
    └─ requirements.txt

## 快速开始（训练）

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    # 图像按 dataset/raw/{Train,Test}/{Fire,Neutral,Smoke} 组织后依次执行：
    python training/01_check_dataset.py
    python training/02_clean_dataset.py
    python training/03_prepare_dataset.py
    python training/04_train_model_v32.py
    python training/05_evaluate_fp32_v32.py
    python training/06_quantize_int8_v32.py
    python training/07_evaluate_int8_v32.py

## 部署到 STM32N647

    # 1. 编译模型代码（stedgeai 环境下）
    stedgeai generate --model fire_smoke_normal_v32_int8.tflite --target stm32n6 --st-neural-art default@user_neuralart.json --input-data-type uint8 --output-data-type float32 --inputs-ch-position chlast
    # 2. 生成权重 hex（加载地址 0x70200000）
    arm-none-eabi-objcopy -I binary network_data.xSPI2.bin --change-addresses 0x70200000 -O ihex ../Binary/network-data.hex
    # 3. CubeIDE 切 Release 编译 Appli（Debug 配置会黑屏）
    # 4. CubeProgrammer 选 ATK 外部加载器，依次下载三件 hex，boot1 拨回 GND 上电

## 硬件平台

正点原子 STM32N647 开发板：STM32N647NIH6（Cortex-M55 @800MHz + Neural-ART NPU @1GHz / 600 GOPS），OV5640 摄像头 + LCD，外部 XSPI NOR Flash。

## 说明

- 原始数据集（3000 张图像）未包含在仓库中，脚本会按上述目录组织自动完成校验、清洗与划分。
- 已知局限：Neutral（正常）与 Smoke（烟雾）两类存在约 21/300 的互相混淆，属该轻量模型固有短板。
