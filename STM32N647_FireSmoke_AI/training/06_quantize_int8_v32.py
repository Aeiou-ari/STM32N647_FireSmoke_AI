"""
06_quantize_int8_v32.py

V3.2 FP32 -> Full Integer INT8 TFLite

Representative Dataset:
    只从 train_split.csv 中选取
    Fire    100
    Neutral 100
    Smoke   100

绝对不使用 Validation / Test 数据进行校准。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. 配置
# ============================================================

IMG_HEIGHT = 160
IMG_WIDTH = 160

SEED = 42

CALIBRATION_PER_CLASS = 100

CLASS_NAMES = [
    "Fire",
    "Neutral",
    "Smoke",
]


# ============================================================
# 2. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "splits"
    / "train_split.csv"
)

FP32_MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "models"
    / "fp32"
    / "fire_smoke_normal_v32_best_loss.keras"
)

INT8_MODEL_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
    / "int8"
)

INT8_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INT8_MODEL_PATH = (
    INT8_MODEL_DIR
    / "fire_smoke_normal_v32_int8.tflite"
)


# ============================================================
# 3. 读取校准图片
#
# 注意：
# 必须与 V3.2 FP32 模型的输入预处理一致：
#
# RGB
#  ↓
# Direct Resize 160x160
#  ↓
# float32 / 255
# ============================================================

def load_calibration_image(
    relative_path
):

    image_path = (
        PROJECT_ROOT
        / Path(relative_path)
    )

    image_data = tf.io.read_file(
        str(image_path)
    )

    image = tf.io.decode_jpeg(
        image_data,
        channels=3
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH],
        method="bilinear"
    )

    image = image / 255.0

    # 增加 Batch 维度
    #
    # [160,160,3]
    # ->
    # [1,160,160,3]

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image.numpy().astype(
        np.float32
    )


# ============================================================
# 4. 选择 Representative Dataset
# ============================================================

def prepare_calibration_samples():

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Train CSV not found: {TRAIN_CSV}"
        )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    samples = []

    print()
    print(
        "Representative Dataset:"
    )

    for label, class_name in enumerate(
        CLASS_NAMES
    ):

        class_df = train_df[
            train_df["label"] == label
        ]

        if len(class_df) < CALIBRATION_PER_CLASS:

            raise ValueError(
                f"{class_name} only contains "
                f"{len(class_df)} images."
            )

        # 固定随机种子，保证实验可复现
        selected = class_df.sample(
            n=CALIBRATION_PER_CLASS,
            random_state=SEED
        )

        print(
            f"  {class_name:<8}: "
            f"{len(selected)}"
        )

        samples.extend(
            selected["filepath"].tolist()
        )

    # 再整体打乱一次
    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(
        samples
    )

    print(
        f"  Total   : {len(samples)}"
    )

    return samples


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "V3.2 FP32 -> Full Integer INT8 Quantization"
    )
    print("=" * 70)

    print()

    print(
        f"TensorFlow : {tf.__version__}"
    )

    print(
        f"FP32 model : {FP32_MODEL_PATH}"
    )

    if not FP32_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"FP32 model not found: "
            f"{FP32_MODEL_PATH}"
        )

    # ========================================================
    # 加载 FP32 模型
    # ========================================================

    model = tf.keras.models.load_model(
        FP32_MODEL_PATH
    )

    print()
    print(
        "[OK] FP32 model loaded."
    )

    # ========================================================
    # 准备 Calibration Dataset
    # ========================================================

    calibration_samples = (
        prepare_calibration_samples()
    )

    # ========================================================
    # Representative Dataset Generator
    # ========================================================

    def representative_dataset():

        for relative_path in calibration_samples:

            image = load_calibration_image(
                relative_path
            )

            yield [
                image
            ]

    # ========================================================
    # TFLite Converter
    # ========================================================

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    # 开启默认优化
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # Representative Dataset
    converter.representative_dataset = (
        representative_dataset
    )

    # ========================================================
    # 强制 Full Integer INT8
    #
    # 不允许模型内部回退到 FP32 算子。
    # ========================================================

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # ========================================================
    # 输入 / 输出均为 INT8
    # ========================================================

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    # ========================================================
    # 转换
    # ========================================================

    print()
    print(
        "Converting model..."
    )

    tflite_model = (
        converter.convert()
    )

    # ========================================================
    # 保存
    # ========================================================

    with open(
        INT8_MODEL_PATH,
        "wb"
    ) as file:

        file.write(
            tflite_model
        )

    model_size = (
        INT8_MODEL_PATH.stat().st_size
    )

    print()
    print("=" * 70)
    print(
        "INT8 QUANTIZATION COMPLETED"
    )
    print("=" * 70)

    print()

    print(
        f"INT8 model:"
    )

    print(
        INT8_MODEL_PATH
    )

    print()

    print(
        f"Model size : "
        f"{model_size / 1024:.2f} KB"
    )

    # ========================================================
    # 检查 TFLite 输入输出
    # ========================================================

    interpreter = tf.lite.Interpreter(
        model_path=str(
            INT8_MODEL_PATH
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    print()
    print(
        "Input tensor:"
    )

    print(
        f"  shape       : "
        f"{input_details[0]['shape']}"
    )

    print(
        f"  dtype       : "
        f"{input_details[0]['dtype']}"
    )

    print(
        f"  scale       : "
        f"{input_details[0]['quantization'][0]}"
    )

    print(
        f"  zero point  : "
        f"{input_details[0]['quantization'][1]}"
    )

    print()

    print(
        "Output tensor:"
    )

    print(
        f"  shape       : "
        f"{output_details[0]['shape']}"
    )

    print(
        f"  dtype       : "
        f"{output_details[0]['dtype']}"
    )

    print(
        f"  scale       : "
        f"{output_details[0]['quantization'][0]}"
    )

    print(
        f"  zero point  : "
        f"{output_details[0]['quantization'][1]}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()