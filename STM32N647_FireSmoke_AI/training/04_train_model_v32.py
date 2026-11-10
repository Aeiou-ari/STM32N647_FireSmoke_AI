"""
04_train_model_v32.py

STM32N647 Fire / Neutral / Smoke V3.2

V3.2 目标：
1. 回到 V2 已验证较稳定的轻量 CNN
2. Direct Resize 160x160
3. 使用较温和的数据增强
4. Label Smoothing = 0.05
5. 每个 Epoch 计算 Validation Confusion Matrix
6. 同时保存：
      best_loss
      best_balanced
7. 特别观察 Neutral <-> Smoke

类别：
    0 = Fire
    1 = Neutral
    2 = Smoke
"""

from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. 基本配置
# ============================================================

SEED = 42

IMG_HEIGHT = 160
IMG_WIDTH = 160
CHANNELS = 3

NUM_CLASSES = 3

BATCH_SIZE = 32
MAX_EPOCHS = 50

INITIAL_LEARNING_RATE = 0.001

LABEL_SMOOTHING = 0.05

CLASS_NAMES = [
    "Fire",
    "Neutral",
    "Smoke",
]


# ============================================================
# 2. 固定随机种子
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. 工程路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "splits"
)

TRAIN_CSV = SPLIT_DIR / "train_split.csv"
VAL_CSV = SPLIT_DIR / "val_split.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_DIR = (
    OUTPUT_DIR
    / "models"
    / "fp32"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. 输出文件
# ============================================================

BEST_LOSS_MODEL_PATH = (
    MODEL_DIR
    / "fire_smoke_normal_v32_best_loss.keras"
)

BEST_BALANCED_MODEL_PATH = (
    MODEL_DIR
    / "fire_smoke_normal_v32_best_balanced.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "fire_smoke_normal_v32_final.keras"
)

HISTORY_PATH = (
    LOG_DIR
    / "fire_smoke_normal_v32_history.csv"
)

VAL_METRICS_PATH = (
    LOG_DIR
    / "fire_smoke_normal_v32_val_metrics.csv"
)

LOSS_CURVE_PATH = (
    FIGURE_DIR
    / "fire_smoke_normal_v32_loss.png"
)

ACCURACY_CURVE_PATH = (
    FIGURE_DIR
    / "fire_smoke_normal_v32_accuracy.png"
)


# ============================================================
# 5. 读取 CSV
# ============================================================

def load_split_csv(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV does not exist: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "filepath",
        "class_name",
        "label",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Invalid CSV format: {csv_path}"
        )

    return df


# ============================================================
# 6. 构建绝对路径
# ============================================================

def build_paths(dataframe):

    return [
        str(
            PROJECT_ROOT
            / Path(relative_path)
        )
        for relative_path
        in dataframe["filepath"]
    ]


# ============================================================
# 7. 图片预处理
#
# V3.2：
#
# JPEG
#   ↓
# RGB
#   ↓
# Direct Resize
#   ↓
# 160x160
#   ↓
# [0,255] -> [0,1]
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

    # --------------------------------------------------------
    # Integer Label -> One-hot Label
    #
    # 0 -> [1,0,0]
    # 1 -> [0,1,0]
    # 2 -> [0,0,1]
    #
    # Label Smoothing 会在 Loss 内部完成。
    # --------------------------------------------------------

    label = tf.one_hot(
        label,
        depth=NUM_CLASSES
    )

    return image, label


# ============================================================
# 8. 温和数据增强
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.03
        ),

        tf.keras.layers.RandomZoom(
            0.08
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

        tf.keras.layers.RandomBrightness(
            0.08,
            value_range=(0.0, 1.0)
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# 9. Dataset
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = build_paths(
        dataframe
    )

    labels = (
        dataframe["label"]
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

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
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
# 10. Conv + BN + ReLU
# ============================================================

def conv_block(
    x,
    filters
):

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.ReLU()(
        x
    )

    return x


# ============================================================
# 11. V3.2 模型
#
# 回到 V2 的轻量结构。
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ),
        name="input"
    )

    # --------------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------------

    x = data_augmentation(
        inputs
    )

    # --------------------------------------------------------
    # Block 1
    # 160 -> 80
    # --------------------------------------------------------

    x = conv_block(
        x,
        16
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # Block 2
    # 80 -> 40
    # --------------------------------------------------------

    x = conv_block(
        x,
        32
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # Block 3
    # 40 -> 20
    # --------------------------------------------------------

    x = conv_block(
        x,
        64
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # Block 4
    # 20 -> 10
    # --------------------------------------------------------

    x = conv_block(
        x,
        96
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # Block 5
    # --------------------------------------------------------

    x = conv_block(
        x,
        128
    )

    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(
        x
    )

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    # --------------------------------------------------------
    # 输出 3 个 logits
    #
    # 0 = Fire
    # 1 = Neutral
    # 2 = Smoke
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        name="output"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="FireSmokeCNN_V32"
    )


# ============================================================
# 12. Validation Balanced Callback
# ============================================================

class ValidationBalancedCallback(
    tf.keras.callbacks.Callback
):

    """
    每个 Epoch 结束后：

    1. 对 Validation Set 推理
    2. 计算 Confusion Matrix
    3. 计算三个类别 Recall
    4. 计算 minimum recall
    5. 统计 Neutral <-> Smoke
    6. 保存 best_balanced 模型

    Balanced Score：

        min(
            Fire Recall,
            Neutral Recall,
            Smoke Recall
        )

    即：优先避免某一个类别明显掉队。

    如果 minimum recall 相同，
    再选择 Neutral/Smoke 双向混淆更少的模型。
    """

    def __init__(
        self,
        validation_dataset,
        validation_labels,
        model_path,
        metrics_path
    ):

        super().__init__()

        self.validation_dataset = (
            validation_dataset
        )

        self.validation_labels = np.asarray(
            validation_labels,
            dtype=np.int32
        )

        self.model_path = model_path
        self.metrics_path = metrics_path

        self.best_score = -1.0
        self.best_ns_errors = float("inf")
        self.best_epoch = None

        self.records = []


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        # ----------------------------------------------------
        # 推理
        # ----------------------------------------------------

        logits = self.model.predict(
            self.validation_dataset,
            verbose=0
        )

        predictions = np.argmax(
            logits,
            axis=1
        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            self.validation_labels,
            predictions,
            labels=[0, 1, 2]
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        row_sum = cm.sum(
            axis=1
        )

        recalls = np.divide(
            np.diag(cm),
            row_sum,
            out=np.zeros(
                NUM_CLASSES,
                dtype=np.float64
            ),
            where=row_sum != 0
        )

        fire_recall = recalls[0]
        neutral_recall = recalls[1]
        smoke_recall = recalls[2]

        # ----------------------------------------------------
        # Balanced Score
        #
        # 最差类别 Recall
        # ----------------------------------------------------

        balanced_score = float(
            np.min(recalls)
        )

        # ----------------------------------------------------
        # Neutral / Smoke 混淆
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 保存记录
        # ----------------------------------------------------

        record = {
            "epoch": epoch + 1,
            "fire_recall": fire_recall,
            "neutral_recall": neutral_recall,
            "smoke_recall": smoke_recall,
            "balanced_score": balanced_score,
            "neutral_to_smoke": neutral_to_smoke,
            "smoke_to_neutral": smoke_to_neutral,
            "ns_errors": ns_errors,
        }

        self.records.append(
            record
        )

        pd.DataFrame(
            self.records
        ).to_csv(
            self.metrics_path,
            index=False
        )

        # ----------------------------------------------------
        # 打印结果
        # ----------------------------------------------------

        print()
        print(
            "Validation class analysis:"
        )

        print(
            cm
        )

        print(
            f"Fire recall    : "
            f"{fire_recall:.4f}"
        )

        print(
            f"Neutral recall : "
            f"{neutral_recall:.4f}"
        )

        print(
            f"Smoke recall   : "
            f"{smoke_recall:.4f}"
        )

        print(
            f"Balanced score : "
            f"{balanced_score:.4f}"
        )

        print(
            f"Neutral->Smoke : "
            f"{neutral_to_smoke}"
        )

        print(
            f"Smoke->Neutral : "
            f"{smoke_to_neutral}"
        )

        # ----------------------------------------------------
        # 是否优于当前 Balanced Model
        # ----------------------------------------------------

        improved = False

        if balanced_score > self.best_score:

            improved = True

        elif np.isclose(
            balanced_score,
            self.best_score
        ):

            if ns_errors < self.best_ns_errors:
                improved = True

        # ----------------------------------------------------
        # 保存模型
        # ----------------------------------------------------

        if improved:

            self.best_score = (
                balanced_score
            )

            self.best_ns_errors = (
                ns_errors
            )

            self.best_epoch = (
                epoch + 1
            )

            self.model.save(
                self.model_path
            )

            print(
                "Saved new best balanced model."
            )


# ============================================================
# 13. 保存训练曲线
# ============================================================

def save_training_curves(
    history
):

    epochs = range(
        1,
        len(history.history["loss"]) + 1
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "V3.2 Training / Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "V3.2 Training / Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        ACCURACY_CURVE_PATH,
        dpi=150
    )

    plt.close()


# ============================================================
# 14. 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "STM32N647 Fire / Neutral / Smoke V3.2 Training"
    )
    print("=" * 70)

    print()

    print(
        f"TensorFlow       : {tf.__version__}"
    )

    print(
        f"Label smoothing  : {LABEL_SMOOTHING}"
    )

    print(
        f"Batch size       : {BATCH_SIZE}"
    )

    print(
        f"Max epochs       : {MAX_EPOCHS}"
    )

    print()

    print("Class mapping:")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index} -> {class_name}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    train_df = load_split_csv(
        TRAIN_CSV
    )

    val_df = load_split_csv(
        VAL_CSV
    )

    print()

    print(
        f"Training images   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation images : "
        f"{len(val_df)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = create_dataset(
        train_df,
        training=True
    )

    val_dataset = create_dataset(
        val_df,
        training=False
    )

    # Validation 的整数标签
    val_labels = (
        val_df["label"]
        .astype(np.int32)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    print()
    model.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LEARNING_RATE
        ),

        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=LABEL_SMOOTHING
        ),

        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    # --------------------------------------------------------
    # Best Loss Model
    # --------------------------------------------------------

    best_loss_checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(

            filepath=BEST_LOSS_MODEL_PATH,

            monitor="val_loss",

            mode="min",

            save_best_only=True,

            verbose=1
        )
    )

    # --------------------------------------------------------
    # Best Balanced Model
    # --------------------------------------------------------

    balanced_callback = (
        ValidationBalancedCallback(

            validation_dataset=val_dataset,

            validation_labels=val_labels,

            model_path=BEST_BALANCED_MODEL_PATH,

            metrics_path=VAL_METRICS_PATH
        )
    )

    # --------------------------------------------------------
    # Reduce LR
    # --------------------------------------------------------

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            mode="min",

            factor=0.5,

            patience=4,

            min_lr=1e-6,

            verbose=1
        )
    )

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            mode="min",

            patience=10,

            restore_best_weights=True,

            verbose=1
        )
    )

    # --------------------------------------------------------
    # CSV Logger
    # --------------------------------------------------------

    csv_logger = (
        tf.keras.callbacks.CSVLogger(
            HISTORY_PATH
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "START V3.2 TRAINING"
    )
    print("=" * 70)

    history = model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=MAX_EPOCHS,

        callbacks=[
            best_loss_checkpoint,
            balanced_callback,
            reduce_lr,
            early_stopping,
            csv_logger,
        ]
    )

    # --------------------------------------------------------
    # 保存 restore_best_weights 后的模型
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )

    # --------------------------------------------------------
    # 曲线
    # --------------------------------------------------------

    save_training_curves(
        history
    )

    # --------------------------------------------------------
    # 结果
    # --------------------------------------------------------

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_loss_epoch = (
        np.argmin(
            history.history["val_loss"]
        )
        + 1
    )

    max_val_accuracy = max(
        history.history["val_accuracy"]
    )

    max_accuracy_epoch = (
        np.argmax(
            history.history["val_accuracy"]
        )
        + 1
    )

    print()
    print("=" * 70)
    print(
        "V3.2 TRAINING COMPLETED"
    )
    print("=" * 70)

    print()

    print(
        f"Best val loss          : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best val loss epoch    : "
        f"{best_loss_epoch}"
    )

    print(
        f"Max val accuracy       : "
        f"{max_val_accuracy:.4%}"
    )

    print(
        f"Max val accuracy epoch : "
        f"{max_accuracy_epoch}"
    )

    print()

    print(
        f"Best balanced score    : "
        f"{balanced_callback.best_score:.4f}"
    )

    print(
        f"Best balanced epoch    : "
        f"{balanced_callback.best_epoch}"
    )

    print(
        f"Best balanced N/S errs : "
        f"{balanced_callback.best_ns_errors}"
    )

    print()

    print("Best loss model:")
    print(BEST_LOSS_MODEL_PATH)

    print()

    print("Best balanced model:")
    print(BEST_BALANCED_MODEL_PATH)

    print()

    print("Final restored model:")
    print(FINAL_MODEL_PATH)

    print()

    print("Validation metrics:")
    print(VAL_METRICS_PATH)

    print()

    print("Training history:")
    print(HISTORY_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()