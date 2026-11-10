"""
02_clean_dataset.py

作用：
1. 从 dataset/raw 读取原始图片
2. 检查图片是否能够正常解码
3. 统一转换为 RGB
4. 统一保存为 JPEG
5. 保留原始 Train / Test 和类别结构

注意：
- 不修改 dataset/raw 中的原始图片
- 本阶段不进行 160x160 Resize
- 几何预处理将在训练阶段统一完成
"""

from pathlib import Path
from PIL import Image, ImageOps
import shutil


# ============================================================
# 1. 工程路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_DIR = DATASET_DIR / "raw"
CLEAN_DIR = DATASET_DIR / "clean"


# ============================================================
# 2. 数据集配置
# ============================================================

SPLITS = [
    "Train",
    "Test",
]

CLASSES = [
    "Fire",
    "Neutral",
    "Smoke",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. 清洗单张图片
# ============================================================

def clean_image(input_path, output_path):
    """
    读取一张图片并进行基本清洗。

    处理：
    1. 根据 EXIF 修正图片方向
    2. 转换成 RGB
    3. 保存为 JPEG

    不进行 Resize。
    """

    with Image.open(input_path) as image:

        # 某些手机/相机照片的旋转方向记录在 EXIF 中，
        # transpose() 会根据 EXIF 自动修正方向。
        image = ImageOps.exif_transpose(image)

        # 无论原图是 RGB、RGBA、L 等格式，
        # 最终统一转换成 3 通道 RGB。
        image = image.convert("RGB")

        # 保存为标准 JPEG。
        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )


# ============================================================
# 4. 清空旧 clean 目录
# ============================================================

def prepare_clean_directory():

    if CLEAN_DIR.exists():

        print()
        print(f"Removing old clean dataset:")
        print(CLEAN_DIR)

        shutil.rmtree(CLEAN_DIR)

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("STM32N647 Dataset Cleaning")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw dataset  : {RAW_DIR}")
    print(f"Clean dataset: {CLEAN_DIR}")

    # --------------------------------------------------------
    # 检查 raw 数据集
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        print()
        print("[ERROR] Raw dataset does not exist:")
        print(RAW_DIR)

        return

    # --------------------------------------------------------
    # 创建干净的输出目录
    # --------------------------------------------------------

    prepare_clean_directory()

    total_input = 0
    total_success = 0
    total_failed = 0

    summary = []

    # --------------------------------------------------------
    # 遍历 Train / Test
    # --------------------------------------------------------

    for split_name in SPLITS:

        for class_name in CLASSES:

            input_dir = (
                RAW_DIR
                / split_name
                / class_name
            )

            output_dir = (
                CLEAN_DIR
                / split_name
                / class_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # -----------------------------------------------
            # 检查输入目录
            # -----------------------------------------------

            if not input_dir.exists():

                print()
                print(
                    f"[ERROR] Directory does not exist: "
                    f"{input_dir}"
                )

                continue

            # -----------------------------------------------
            # 找到所有支持的图片
            # -----------------------------------------------

            image_files = [
                path
                for path in input_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ]

            image_files.sort()

            success = 0
            failed = 0

            print()
            print("-" * 70)
            print(
                f"Cleaning {split_name}/{class_name}"
                f" : {len(image_files)} images"
            )
            print("-" * 70)

            # -----------------------------------------------
            # 清洗每一张图片
            # -----------------------------------------------

            for index, input_path in enumerate(
                image_files,
                start=1
            ):

                # 使用统一编号，避免不同扩展名造成重名
                output_name = (
                    f"{class_name.lower()}_"
                    f"{index:04d}.jpg"
                )

                output_path = (
                    output_dir
                    / output_name
                )

                try:

                    clean_image(
                        input_path,
                        output_path
                    )

                    success += 1

                except Exception as error:

                    failed += 1

                    print(
                        f"[FAILED] {input_path.name}"
                    )

                    print(
                        f"         {error}"
                    )

            total_input += len(image_files)
            total_success += success
            total_failed += failed

            summary.append(
                (
                    split_name,
                    class_name,
                    len(image_files),
                    success,
                    failed,
                )
            )

    # ========================================================
    # 6. 输出汇总
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'Split':<10}"
        f"{'Class':<12}"
        f"{'Input':>10}"
        f"{'Success':>10}"
        f"{'Failed':>10}"
    )

    print("-" * 52)

    for (
        split_name,
        class_name,
        input_count,
        success,
        failed,
    ) in summary:

        print(
            f"{split_name:<10}"
            f"{class_name:<12}"
            f"{input_count:>10}"
            f"{success:>10}"
            f"{failed:>10}"
        )

    print("-" * 52)

    print(
        f"{'TOTAL':<22}"
        f"{total_input:>10}"
        f"{total_success:>10}"
        f"{total_failed:>10}"
    )

    # ========================================================
    # 7. 最终结果
    # ========================================================

    print()
    print("=" * 70)

    if total_failed == 0:

        print("[OK] Dataset cleaning completed successfully.")

    else:

        print(
            f"[WARNING] {total_failed} image(s) failed "
            f"during cleaning."
        )

    print()
    print(f"Input images : {total_input}")
    print(f"Clean images : {total_success}")
    print(f"Failed       : {total_failed}")

    print()
    print(f"Output:")
    print(CLEAN_DIR)

    print("=" * 70)


# ============================================================
# 8. 程序入口
# ============================================================

if __name__ == "__main__":
    main()