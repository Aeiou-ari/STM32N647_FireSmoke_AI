"""
05_evaluate_fp32_v32.py

同时评估 V3.2：

1. best_loss 模型
2. best_balanced 模型

使用完全独立的 Test Set：

    Fire    100
    Neutral 100
    Smoke   100

重点观察：

    Accuracy
    Precision
    Recall
    F1-score

    Neutral -> Smoke
    Smoke -> Neutral
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
CHANNELS = 3

BATCH_SIZE = 32

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

MODEL_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
    / "fp32"
)

BEST_LOSS_MODEL = (
    MODEL_DIR
    / "fire_smoke_normal_v32_best_loss.keras"
)

BEST_BALANCED_MODEL = (
    MODEL_DIR
    / "fire_smoke_normal_v32_best_balanced.keras"
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


# ============================================================
# 3. 图片预处理
#
# 必须和 V3.2 Training 完全一致：
#
# RGB
# ↓
# Direct Resize 160x160
# ↓
# /255
#
# 注意：
# Evaluation 不使用 Data Augmentation。
# ============================================================

def load_and_preprocess_image(
    image_path,
    label
):

    image_data = tf.io.read_file(
        image_path
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

    image.set_shape(
        [
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ]
    )

    return image, label


# ============================================================
# 4. Test Dataset
# ============================================================

def create_test_dataset(
    test_df
):

    paths = [
        str(
            PROJECT_ROOT
            / Path(relative_path)
        )
        for relative_path
        in test_df["filepath"]
    ]

    labels = (
        test_df["label"]
        .astype(np.int32)
        .to_numpy()
    )

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels
            )
        )
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 5. 评估一个模型
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    test_dataset,
    true_labels
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 70)

    print()

    print("Model:")
    print(model_path)

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model does not exist: {model_path}"
        )

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path
    )

    # --------------------------------------------------------
    # 预测 logits
    # --------------------------------------------------------

    logits = model.predict(
        test_dataset,
        verbose=1
    )

    predictions = np.argmax(
        logits,
        axis=1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # 专项指标
    # --------------------------------------------------------

    fire_correct = int(
        cm[0, 0]
    )

    neutral_correct = int(
        cm[1, 1]
    )

    smoke_correct = int(
        cm[2, 2]
    )

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

    recalls = np.diag(cm) / cm.sum(
        axis=1
    )

    minimum_recall = float(
        np.min(recalls)
    )

    # ========================================================
    # 输出
    # ========================================================

    print()
    print(
        f"{model_name} TEST RESULTS"
    )

    print("-" * 70)

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
        f"{fire_correct}/100"
    )

    print(
        f"Neutral correct    : "
        f"{neutral_correct}/100"
    )

    print(
        f"Smoke correct      : "
        f"{smoke_correct}/100"
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
        f"Minimum recall     : "
        f"{minimum_recall:.4%}"
    )

    # ========================================================
    # 保存报告
    # ========================================================

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    report_path = (
        REPORT_DIR
        / f"fire_smoke_normal_v32_{safe_name}_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{model_name} TEST RESULTS\n"
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

        file.write(report)

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
            f"Fire correct     : "
            f"{fire_correct}/100\n"
        )

        file.write(
            f"Neutral correct  : "
            f"{neutral_correct}/100\n"
        )

        file.write(
            f"Smoke correct    : "
            f"{smoke_correct}/100\n"
        )

        file.write(
            f"Neutral -> Smoke : "
            f"{neutral_to_smoke}/100\n"
        )

        file.write(
            f"Smoke -> Neutral : "
            f"{smoke_to_neutral}/100\n"
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
    # Confusion Matrix 图片
    # ========================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"V3.2 {model_name} Confusion Matrix"
    )

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR
        / f"fire_smoke_normal_v32_{safe_name}_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=150
    )

    plt.close()

    return {
        "name": model_name,
        "accuracy": accuracy,
        "fire_recall": recalls[0],
        "neutral_recall": recalls[1],
        "smoke_recall": recalls[2],
        "minimum_recall": minimum_recall,
        "neutral_to_smoke": neutral_to_smoke,
        "smoke_to_neutral": smoke_to_neutral,
        "ns_errors": ns_errors,
    }


# ============================================================
# 6. 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "STM32N647 V3.2 FP32 Evaluation"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Test CSV
    # --------------------------------------------------------

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV does not exist: {TEST_CSV}"
        )

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

    print(
        "Test class distribution:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                true_labels == index
            )
        )

        print(
            f"  {class_name:<8}: {count}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = create_test_dataset(
        test_df
    )

    # --------------------------------------------------------
    # Best Loss
    # --------------------------------------------------------

    result_loss = evaluate_model(

        model_name="Best Loss",

        model_path=BEST_LOSS_MODEL,

        test_dataset=test_dataset,

        true_labels=true_labels
    )

    # --------------------------------------------------------
    # Best Balanced
    # --------------------------------------------------------

    result_balanced = evaluate_model(

        model_name="Best Balanced",

        model_path=BEST_BALANCED_MODEL,

        test_dataset=test_dataset,

        true_labels=true_labels
    )

    # ========================================================
    # 7. 最终对比
    # ========================================================

    print()
    print()
    print("=" * 90)
    print(
        "V3.2 MODEL COMPARISON"
    )
    print("=" * 90)

    print()

    print(
        f"{'Model':<18}"
        f"{'Accuracy':>12}"
        f"{'Fire R':>10}"
        f"{'Neutral R':>12}"
        f"{'Smoke R':>10}"
        f"{'Min R':>10}"
        f"{'N->S':>8}"
        f"{'S->N':>8}"
    )

    print("-" * 88)

    for result in [
        result_loss,
        result_balanced
    ]:

        print(
            f"{result['name']:<18}"
            f"{result['accuracy']:>11.2%}"
            f"{result['fire_recall']:>10.2%}"
            f"{result['neutral_recall']:>12.2%}"
            f"{result['smoke_recall']:>10.2%}"
            f"{result['minimum_recall']:>10.2%}"
            f"{result['neutral_to_smoke']:>8}"
            f"{result['smoke_to_neutral']:>8}"
        )

    print("-" * 88)

    print()
    print(
        "[OK] V3.2 FP32 evaluation completed."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()