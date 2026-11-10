"""
01_check_dataset.py

作用：
1. 检查原始数据集目录是否正确
2. 统计每个类别的图片数量
3. 检查图片是否能够正常读取
4. 检查图片的尺寸和颜色通道
5. 统计图片格式
6. 输出最终检查结果

注意：
本脚本只读取图片，不会修改原始数据。
"""

from pathlib import Path
from collections import Counter

from PIL import Image


# ============================================================
# 1. 工程路径
# ============================================================

# 当前脚本位置：
# D:/STM32N647_FIRE/scripts/01_check_dataset.py
#
# parent        -> scripts
# parent.parent -> STM32N647_FIRE
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"

TRAIN_DIR = RAW_DIR / "Train"
TEST_DIR = RAW_DIR / "Test"


# ============================================================
# 2. 数据集类别
# ============================================================

CLASSES = [
    "Fire",
    "Neutral",
    "Smoke",
]


# ============================================================
# 3. 支持的图片格式
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 4. 检查单张图片
# ============================================================

def check_image(image_path):
    """
    检查单张图片。

    返回：
        width       图片宽度
        height      图片高度
        mode        图片颜色模式，例如 RGB
        image_format 图片文件格式，例如 JPEG
    """

    with Image.open(image_path) as image:

        # verify() 用于检查图片文件是否损坏
        image.verify()

    # verify() 后需要重新打开图片
    with Image.open(image_path) as image:

        width, height = image.size
        mode = image.mode
        image_format = image.format

    return width, height, mode, image_format


# ============================================================
# 5. 检查一个类别目录
# ============================================================

def check_class_directory(class_dir):
    """
    检查某一个类别目录中的全部图片。
    """

    result = {
        "total": 0,
        "valid": 0,
        "broken": 0,
        "non_rgb": 0,
        "extensions": Counter(),
        "modes": Counter(),
        "sizes": Counter(),
        "broken_files": [],
    }

    # --------------------------------------------------------
    # 检查目录是否存在
    # --------------------------------------------------------

    if not class_dir.exists():
        print(f"[ERROR] Directory does not exist: {class_dir}")
        return result

    # --------------------------------------------------------
    # 获取目录中的图片
    # --------------------------------------------------------

    image_files = []

    for file_path in class_dir.iterdir():

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension in SUPPORTED_EXTENSIONS:
            image_files.append(file_path)

    image_files.sort()

    result["total"] = len(image_files)

    # --------------------------------------------------------
    # 逐张检查
    # --------------------------------------------------------

    for image_path in image_files:

        extension = image_path.suffix.lower()

        result["extensions"][extension] += 1

        try:

            width, height, mode, image_format = check_image(image_path)

            result["valid"] += 1

            result["modes"][mode] += 1

            result["sizes"][(width, height)] += 1

            if mode != "RGB":
                result["non_rgb"] += 1

        except Exception as error:

            result["broken"] += 1

            result["broken_files"].append(
                (image_path, str(error))
            )

    return result


# ============================================================
# 6. 打印类别检查结果
# ============================================================

def print_class_result(split_name, class_name, result):

    print()
    print("-" * 70)

    print(f"Split : {split_name}")
    print(f"Class : {class_name}")

    print("-" * 70)

    print(f"Total images : {result['total']}")
    print(f"Valid images : {result['valid']}")
    print(f"Broken images: {result['broken']}")
    print(f"Non-RGB      : {result['non_rgb']}")

    print()

    print("File extensions:")

    for extension, count in result["extensions"].most_common():
        print(f"  {extension:<8} : {count}")

    print()

    print("Image modes:")

    for mode, count in result["modes"].most_common():
        print(f"  {mode:<8} : {count}")

    print()

    print("Most common image sizes:")

    # 只显示数量最多的前 10 种尺寸
    for (width, height), count in result["sizes"].most_common(10):

        print(
            f"  {width:>5} x {height:<5} : {count}"
        )

    # --------------------------------------------------------
    # 损坏文件
    # --------------------------------------------------------

    if result["broken_files"]:

        print()
        print("Broken files:")

        for file_path, error in result["broken_files"]:

            print(f"  {file_path}")
            print(f"    Error: {error}")


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("STM32N647 Fire / Neutral / Smoke Dataset Checker")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {RAW_DIR}")

    # --------------------------------------------------------
    # 检查数据集根目录
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        print()
        print("[ERROR] Dataset directory does not exist:")
        print(RAW_DIR)

        return

    # --------------------------------------------------------
    # 数据集划分
    # --------------------------------------------------------

    splits = {
        "Train": TRAIN_DIR,
        "Test": TEST_DIR,
    }

    total_images = 0
    total_valid = 0
    total_broken = 0
    total_non_rgb = 0

    # 用于最后打印汇总表
    summary = []

    # --------------------------------------------------------
    # 检查 Train / Test
    # --------------------------------------------------------

    for split_name, split_dir in splits.items():

        for class_name in CLASSES:

            class_dir = split_dir / class_name

            result = check_class_directory(class_dir)

            print_class_result(
                split_name,
                class_name,
                result
            )

            total_images += result["total"]
            total_valid += result["valid"]
            total_broken += result["broken"]
            total_non_rgb += result["non_rgb"]

            summary.append(
                (
                    split_name,
                    class_name,
                    result["total"],
                    result["valid"],
                    result["broken"],
                    result["non_rgb"],
                )
            )

    # ========================================================
    # 8. 最终汇总
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'Split':<10}"
        f"{'Class':<12}"
        f"{'Total':>10}"
        f"{'Valid':>10}"
        f"{'Broken':>10}"
        f"{'Non-RGB':>10}"
    )

    print("-" * 62)

    for (
        split_name,
        class_name,
        total,
        valid,
        broken,
        non_rgb,
    ) in summary:

        print(
            f"{split_name:<10}"
            f"{class_name:<12}"
            f"{total:>10}"
            f"{valid:>10}"
            f"{broken:>10}"
            f"{non_rgb:>10}"
        )

    print("-" * 62)

    print(
        f"{'TOTAL':<22}"
        f"{total_images:>10}"
        f"{total_valid:>10}"
        f"{total_broken:>10}"
        f"{total_non_rgb:>10}"
    )

    # ========================================================
    # 9. 检查结论
    # ========================================================

    print()
    print("=" * 70)

    if total_broken == 0:

        print("[OK] No broken images found.")

    else:

        print(
            f"[WARNING] Found {total_broken} broken image(s)."
        )

    if total_non_rgb == 0:

        print("[OK] All valid images are RGB.")

    else:

        print(
            f"[INFO] Found {total_non_rgb} non-RGB image(s)."
        )

        print(
            "They will be converted to RGB during dataset cleaning."
        )

    print()

    print(f"Total images checked : {total_images}")

    print("=" * 70)


# ============================================================
# 10. 程序入口
# ============================================================

if __name__ == "__main__":
    main()