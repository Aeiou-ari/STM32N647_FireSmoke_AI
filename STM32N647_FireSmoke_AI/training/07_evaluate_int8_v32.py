"""
07_evaluate_int8_v32.py

评估 V3.2 Full Integer INT8 TFLite。

测试集：
    Fire    100
    Neutral 100
    Smoke   100

测试集完全独立，不参与训练和 INT8 Calibration。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. 配置
# ============================================================

IMG_HEIGHT = 160
IMG_WIDTH = 160

CLASS_NAMES = [
    "Fire",
    "Neutral",
    "Smoke",
]


# ============================================================
# 2. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "splits"
    / "test_split.csv"
)

INT8_MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "models"
    / "int8"
    / "fire_smoke_normal_v32_int8.tflite"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "output"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "output"
    / "figures"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_PATH = (
    REPORT_DIR
    / "fire_smoke_normal_v32_int8_report.txt"
)

CONFUSION_MATRIX_PATH = (
    FIGURE_DIR
    / "fire_smoke_normal_v32_int8_confusion_matrix.png"
)


# ============================================================
# 3. FP32 输入预处理
#
# 先生成模型训练时的真实输入：
#
# RGB
# ↓
# Direct Resize
# ↓
# /255
# ↓
# float32 [0,1]
#
# 后面再根据 TFLite Input Quantization 参数转换成 int8。
# ============================================================

def load_image(
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

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image.numpy().astype(
        np.float32
    )


# ============================================================
# 4. Float -> INT8
#
# TFLite Quantization:
#
# real_value =
#     scale * (quantized_value - zero_point)
#
# 因此：
#
# quantized_value =
#     real_value / scale + zero_point
# ============================================================

def quantize_input(
    image,
    scale,
    zero_point
):

    if scale == 0:

        raise ValueError(
            "Invalid input quantization scale = 0"
        )

    quantized = (
        np.round(
            image / scale
            + zero_point
        )
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    )

    return quantized.astype(
        np.int8
    )


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "V3.2 INT8 TEST RESULTS"
    )
    print("=" * 70)

    if not INT8_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"INT8 model not found: "
            f"{INT8_MODEL_PATH}"
        )

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found: "
            f"{TEST_CSV}"
        )

    # ========================================================
    # Test CSV
    # ========================================================

    test_df = pd.read_csv(
        TEST_CSV
    )

    true_labels = (
        test_df["label"]
        .astype(np.int32)
        .to_numpy()
    )

    print()
    print(
        f"Test images : {len(test_df)}"
    )

    print()

    for label, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                true_labels == label
            )
        )

        print(
            f"  {class_name:<8}: {count}"
        )

    # ========================================================
    # TFLite Interpreter
    # ========================================================

    interpreter = tf.lite.Interpreter(
        model_path=str(
            INT8_MODEL_PATH
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    input_index = (
        input_details["index"]
    )

    output_index = (
        output_details["index"]
    )

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    print()
    print(
        "Input tensor:"
    )

    print(
        f"  shape      : "
        f"{input_details['shape']}"
    )

    print(
        f"  dtype      : "
        f"{input_details['dtype']}"
    )

    print(
        f"  scale      : "
        f"{input_scale}"
    )

    print(
        f"  zero point : "
        f"{input_zero_point}"
    )

    print()

    print(
        "Output tensor:"
    )

    print(
        f"  shape      : "
        f"{output_details['shape']}"
    )

    print(
        f"  dtype      : "
        f"{output_details['dtype']}"
    )

    print(
        f"  scale      : "
        f"{output_scale}"
    )

    print(
        f"  zero point : "
        f"{output_zero_point}"
    )

    # ========================================================
    # 推理
    # ========================================================

    predictions = []

    print()
    print(
        "Running INT8 inference..."
    )

    for index, row in test_df.iterrows():

        image = load_image(
            row["filepath"]
        )

        input_int8 = quantize_input(
            image,
            input_scale,
            input_zero_point
        )

        interpreter.set_tensor(
            input_index,
            input_int8
        )

        interpreter.invoke()

        output_int8 = interpreter.get_tensor(
            output_index
        )

        # ----------------------------------------------------
        # 对于分类 argmax：
        #
        # 如果三个输出共享相同 scale / zero point，
        # 直接对 INT8 输出做 argmax 即可。
        #
        # 这里仍然进行反量化，便于后续查看真实 logits。
        # ----------------------------------------------------

        output_float = (
            output_scale
            * (
                output_int8.astype(
                    np.float32
                )
                - output_zero_point
            )
        )

        prediction = int(
            np.argmax(
                output_float[0]
            )
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 50 == 0
            or
            (index + 1) == len(test_df)
        ):

            print(
                f"  {index + 1}"
                f"/{len(test_df)}"
            )

    predictions = np.asarray(
        predictions,
        dtype=np.int32
    )

    # ========================================================
    # Accuracy
    # ========================================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    # ========================================================
    # Classification Report
    # ========================================================

    report = classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    # ========================================================
    # Recall
    # ========================================================

    recalls = (
        np.diag(cm)
        / cm.sum(axis=1)
    )

    fire_recall = recalls[0]
    neutral_recall = recalls[1]
    smoke_recall = recalls[2]

    minimum_recall = float(
        np.min(recalls)
    )

    # ========================================================
    # Neutral / Smoke
    # ========================================================

    neutral_to_smoke = int(
        cm[1, 2]
    )

    smoke_to_neutral = int(
        cm[2, 1]
    )

    ns_errors = (
        neutral_to_smoke
        + smoke_to_neutral
    )

    # ========================================================
    # 打印结果
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V3.2 INT8 TEST RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Test accuracy : "
        f"{accuracy:.4%}"
    )

    print()

    print(
        "Classification report:"
    )

    print()
    print(report)

    print(
        "Confusion matrix:"
    )

    print()
    print(cm)

    print()

    print(
        "NEUTRAL / SMOKE ANALYSIS"
    )

    print("-" * 70)

    print(
        f"Fire correct       : "
        f"{cm[0, 0]}/100"
    )

    print(
        f"Neutral correct    : "
        f"{cm[1, 1]}/100"
    )

    print(
        f"Smoke correct      : "
        f"{cm[2, 2]}/100"
    )

    print()

    print(
        f"Neutral -> Smoke   : "
        f"{neutral_to_smoke}/100"
    )

    print(
        f"Smoke -> Neutral   : "
        f"{smoke_to_neutral}/100"
    )

    print(
        f"N/S total errors   : "
        f"{ns_errors}"
    )

    print()

    print(
        f"Fire recall        : "
        f"{fire_recall:.2%}"
    )

    print(
        f"Neutral recall     : "
        f"{neutral_recall:.2%}"
    )

    print(
        f"Smoke recall       : "
        f"{smoke_recall:.2%}"
    )

    print(
        f"Minimum recall     : "
        f"{minimum_recall:.2%}"
    )

    # ========================================================
    # 保存 TXT Report
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "V3.2 INT8 TEST RESULTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.4%}\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            report
        )

        file.write(
            "\nConfusion Matrix\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            np.array2string(cm)
        )

        file.write(
            "\n\nNeutral / Smoke Analysis\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Neutral -> Smoke : "
            f"{neutral_to_smoke}\n"
        )

        file.write(
            f"Smoke -> Neutral : "
            f"{smoke_to_neutral}\n"
        )

        file.write(
            f"N/S total errors : "
            f"{ns_errors}\n"
        )

        file.write(
            f"Minimum recall   : "
            f"{minimum_recall:.4%}\n"
        )

    # ========================================================
    # 保存 Confusion Matrix
    # ========================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "V3.2 INT8 Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    print()
    print(
        "Report:"
    )
    print(
        REPORT_PATH
    )

    print()
    print(
        "Confusion matrix figure:"
    )
    print(
        CONFUSION_MATRIX_PATH
    )

    print()
    print(
        "[OK] V3.2 INT8 evaluation completed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()