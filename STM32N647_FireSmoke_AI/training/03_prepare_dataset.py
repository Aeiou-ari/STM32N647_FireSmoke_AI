"""
03_prepare_dataset.py

作用：
1. 从 dataset/clean 读取清洗后的数据
2. 将原始 Train 按类别拆分为：
      90% Training
      10% Validation
3. 保留原始 Test 作为独立测试集
4. 生成：
      train_split.csv
      val_split.csv
      test_split.csv
5. 固定随机种子，使数据划分可以复现

注意：
Test 数据不会参与训练和 Validation。
"""

from pathlib import Path
import random

import pandas as pd


# ============================================================
# 1. 工程路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

CLEAN_DIR = DATASET_DIR / "clean"
SPLIT_DIR = DATASET_DIR / "splits"

TRAIN_DIR = CLEAN_DIR / "Train"
TEST_DIR = CLEAN_DIR / "Test"


# ============================================================
# 2. 数据集配置
# ============================================================

CLASSES = [
    "Fire",
    "Neutral",
    "Smoke",
]

# 类别编号固定下来。
#
# 这个顺序非常重要：
#
# 0 = Fire
# 1 = Neutral
# 2 = Smoke
#
# 后续：
# TensorFlow
# TFLite
# ST Edge AI
# STM32N647
#
# 都必须保持这个类别顺序。
CLASS_TO_INDEX = {
    "Fire": 0,
    "Neutral": 1,
    "Smoke": 2,
}


# Validation 占原始 Train 的 10%
VALIDATION_RATIO = 0.10


# 固定随机种子
#
# 只要：
# 1. 数据集没有变化
# 2. RANDOM_SEED 没有变化
#
# 每次运行得到的 Train / Validation 划分都相同。
RANDOM_SEED = 42


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. 获取图片
# ============================================================

def get_images(directory):

    images = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    images.sort()

    return images


# ============================================================
# 4. 将绝对路径转换成相对工程路径
# ============================================================

def make_relative_path(path):

    """
    CSV 中不保存：

    D:/STM32N647_FIRE/...

    而保存：

    dataset/clean/Train/Fire/fire_0001.jpg

    这样把整个工程复制到其他电脑以后，
    CSV 仍然可以正常使用。
    """

    relative_path = path.relative_to(PROJECT_ROOT)

    # 使用 POSIX "/" 格式保存，
    # 避免 Windows 反斜杠带来的转义问题。
    return relative_path.as_posix()


# ============================================================
# 5. 创建一条 CSV 记录
# ============================================================

def create_record(image_path, class_name):

    return {
        "filepath": make_relative_path(image_path),
        "class_name": class_name,
        "label": CLASS_TO_INDEX[class_name],
    }


# ============================================================
# 6. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("STM32N647 Dataset Split Preparation")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Clean dataset: {CLEAN_DIR}")
    print(f"Split output : {SPLIT_DIR}")

    # --------------------------------------------------------
    # 检查数据集
    # --------------------------------------------------------

    if not CLEAN_DIR.exists():

        print()
        print("[ERROR] Clean dataset does not exist:")
        print(CLEAN_DIR)

        return

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 独立随机数生成器
    # --------------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    train_records = []
    val_records = []
    test_records = []

    summary = []

    # ========================================================
    # 7. Train → Training + Validation
    # ========================================================

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Directory does not exist: {class_dir}"
            )

        images = get_images(class_dir)

        # ----------------------------------------------------
        # 打乱图片顺序
        # ----------------------------------------------------

        rng.shuffle(images)

        # ----------------------------------------------------
        # 计算 Validation 数量
        # ----------------------------------------------------

        val_count = int(
            len(images) * VALIDATION_RATIO
        )

        # ----------------------------------------------------
        # 划分
        # ----------------------------------------------------

        val_images = images[:val_count]
        train_images = images[val_count:]

        # ----------------------------------------------------
        # 创建记录
        # ----------------------------------------------------

        for image_path in train_images:

            train_records.append(
                create_record(
                    image_path,
                    class_name
                )
            )

        for image_path in val_images:

            val_records.append(
                create_record(
                    image_path,
                    class_name
                )
            )

        summary.append(
            (
                class_name,
                len(train_images),
                len(val_images),
            )
        )

    # ========================================================
    # 8. 原始 Test 保持独立
    # ========================================================

    for class_name in CLASSES:

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Directory does not exist: {class_dir}"
            )

        images = get_images(class_dir)

        for image_path in images:

            test_records.append(
                create_record(
                    image_path,
                    class_name
                )
            )

    # ========================================================
    # 9. 再打乱 CSV 中记录顺序
    # ========================================================

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    # ========================================================
    # 10. 创建 DataFrame
    # ========================================================

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)

    # ========================================================
    # 11. 保存 CSV
    # ========================================================

    train_csv = SPLIT_DIR / "train_split.csv"
    val_csv = SPLIT_DIR / "val_split.csv"
    test_csv = SPLIT_DIR / "test_split.csv"

    train_df.to_csv(
        train_csv,
        index=False,
        encoding="utf-8"
    )

    val_df.to_csv(
        val_csv,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        test_csv,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # 12. 输出类别统计
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("DATASET SPLIT SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'Class':<15}"
        f"{'Train':>12}"
        f"{'Validation':>15}"
        f"{'Test':>12}"
    )

    print("-" * 54)

    for (
        class_name,
        train_count,
        val_count,
    ) in summary:

        test_count = len(
            test_df[
                test_df["class_name"] == class_name
            ]
        )

        print(
            f"{class_name:<15}"
            f"{train_count:>12}"
            f"{val_count:>15}"
            f"{test_count:>12}"
        )

    print("-" * 54)

    print(
        f"{'TOTAL':<15}"
        f"{len(train_df):>12}"
        f"{len(val_df):>15}"
        f"{len(test_df):>12}"
    )

    # ========================================================
    # 13. 类别编号
    # ========================================================

    print()
    print("Class mapping:")

    for class_name in CLASSES:

        print(
            f"  {CLASS_TO_INDEX[class_name]}"
            f" -> {class_name}"
        )

    # ========================================================
    # 14. 文件位置
    # ========================================================

    print()
    print("Generated files:")

    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print(f"  {test_csv}")

    print()

    print(
        f"Random seed      : {RANDOM_SEED}"
    )

    print(
        f"Validation ratio : "
        f"{VALIDATION_RATIO:.0%}"
    )

    print()

    print(
        "[OK] Dataset split preparation "
        "completed successfully."
    )

    print("=" * 70)


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":
    main()