#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D LUT Creator - Stepwise Processing Version
逐张图片处理3D LUT生成器，避免大数据集内存问题

Usage:
    python main_stepwise.py --photoa photoa_folder --photob photob_folder --output output_lut

Features:
    - 逐张图片处理，避免内存溢出
    - 实时显示处理进度
    - 自动内存管理
    - 支持大数据集 (百万级颜色映射)
"""

import os
import sys
import argparse
import time
from pathlib import Path

from lut_generator_stepwise import LUT3DGeneratorStepwise
from lut_exporter import LUTExporter


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create 3D Lookup Tables (LUT) from image pairs with stepwise processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_stepwise.py --photoa photoa --photob photob --output my_lut
  python main_stepwise.py --photoa ./input/a --photob ./input/b --output ./output/lut --size 64 --preview
        """
    )

    parser.add_argument(
        '--photoa',
        type=str,
        required=True,
        help='Base RGB images folder path (input colors)'
    )

    parser.add_argument(
        '--photob',
        type=str,
        required=True,
        help='Mapped RGB images folder path (output colors)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output LUT file base name (without extension)'
    )

    parser.add_argument(
        '--size',
        type=int,
        default=64,
        help='LUT grid size (default: 64)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='Device to use for computation (default: auto)'
    )

    parser.add_argument(
        '--formats',
        type=str,
        default='cube,plut',
        help='Export formats: cube,3dl,dat,npy,plut,all (default: cube,plut)'
    )

    parser.add_argument(
        '--title',
        type=str,
        default='3D LUT from Image Pairs',
        help='LUT file title'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads for parallel image processing (default: 8)'
    )

    parser.add_argument(
        '--depth',
        type=int,
        default=8,
        choices=[8, 16],
        help='Input image bit depth (8 or 16, default: 8)'
    )

    return parser.parse_args()


def validate_input(photoa_dir: str, photob_dir: str):
    """Validate input directories and files"""
    if not os.path.exists(photoa_dir):
        raise ValueError(f"photoa directory does not exist: {photoa_dir}")

    if not os.path.exists(photob_dir):
        raise ValueError(f"photob directory does not exist: {photob_dir}")

    if not os.path.isdir(photoa_dir):
        raise ValueError(f"photoa path is not a directory: {photoa_dir}")

    if not os.path.isdir(photob_dir):
        raise ValueError(f"photob path is not a directory: {photob_dir}")

    # Check for image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    photoa_files = [f for f in os.listdir(photoa_dir)
                   if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(photoa_dir, f))]

    photob_files = [f for f in os.listdir(photob_dir)
                   if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(photob_dir, f))]

    if not photoa_files:
        raise ValueError(f"No image files found in photoa directory")

    if not photob_files:
        raise ValueError(f"No image files found in photob directory")

    # Count matching pairs
    matching_pairs = 0
    for photoa_file in photoa_files:
        photob_path = os.path.join(photob_dir, photoa_file)
        if os.path.exists(photob_path):
            matching_pairs += 1

    print(f"Found {len(photoa_files)} files in photoa directory")
    print(f"Found {len(photob_files)} files in photob directory")
    print(f"Matching pairs: {matching_pairs}")

    return True


def main():
    """Main program"""
    print("=" * 70)
    print("3D LUT Creator - Stepwise Processing Version")
    print("逐张图片处理，解决大数据集内存问题")
    print("=" * 70)

    try:
        # Parse arguments
        args = parse_arguments()

        # Validate input
        validate_input(args.photoa, args.photob)

        print(f"\n配置信息:")
        print(f"  基准图片目录: {args.photoa}")
        print(f"  映射图片目录: {args.photob}")
        print(f"  输出文件: {args.output}")
        print(f"  LUT尺寸: {args.size}")
        print(f"  并行线程数: {args.threads}")
        print(f"  导出格式: {args.formats}")
        print(f"  LUT标题: {args.title}")
        print(f"  位深支持: {args.depth}-bit")

        # Step 1: Generate 3D LUT using stepwise processing
        print(f"\n{'='*30} 生成3D LUT {'='*30}")

        start_time = time.time()

        generator = LUT3DGeneratorStepwise(lut_size=args.size, device=args.device, bit_depth=args.depth)
        lut_data = generator.generate_3d_lut_stepwise(args.photoa, args.photob, num_threads=args.threads)

        generation_time = time.time() - start_time

        print(f"✅ LUT生成完成!")
        print(f"   数据形状: {lut_data.shape}")
        print(f"   生成时间: {generation_time:.2f}秒")
        print(f"   内存使用: {lut_data.nbytes / 1024 / 1024:.1f} MB")

        # Step 2: Export LUT files
        print(f"\n{'='*30} 导出LUT文件 {'='*30}")

        # Create a wrapper for the generator to use with exporter
        class GeneratorWrapper:
            def __init__(self, lut_data, lut_size, bit_depth):
                self.lut_data = lut_data
                self.lut_size = lut_size
                self.bit_depth = bit_depth

        wrapper = GeneratorWrapper(lut_data, generator.lut_size, generator.bit_depth)
        exporter = LUTExporter(wrapper)

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        os.makedirs(output_dir, exist_ok=True)

        # Export based on specified formats
        formats = args.formats.lower().split(',') if args.formats != 'all' else ['cube', '3dl', 'dat', 'npy', 'plut']

        exported_files = []
        export_start = time.time()

        for fmt in formats:
            fmt = fmt.strip()
            try:
                if fmt == 'cube':
                    filename = f"{args.output}.cube"
                    exporter.export_cube(filename, args.title)
                    exported_files.append(filename)
                    print(f"  ✓ {filename}")

                elif fmt == '3dl':
                    filename = f"{args.output}.3dl"
                    exporter.export_3dl(filename, args.title)
                    exported_files.append(filename)
                    print(f"  ✓ {filename}")

                elif fmt == 'dat':
                    filename = f"{args.output}.dat"
                    exporter.export_dat(filename, args.title)
                    exported_files.append(filename)
                    print(f"  ✓ {filename}")

                elif fmt == 'npy':
                    filename = f"{args.output}.npy"
                    exporter.export_numpy(filename)
                    exported_files.append(filename)
                    print(f"  ✓ {filename}")

                elif fmt == 'plut':
                    filename = f"{args.output}.plut"
                    exporter.export_plut(filename)
                    exported_files.append(filename)
                    print(f"  ✓ {filename}")

                else:
                    print(f"  ⚠ 跳过不支持的格式: {fmt}")

            except Exception as e:
                print(f"  ❌ 导出{fmt}格式失败: {e}")

        export_time = time.time() - export_start

        # Completion
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("🎉 3D LUT分步生成完成!")
        print(f"{'='*70}")

        print(f"📊 性能统计:")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   LUT生成: {generation_time:.2f}秒 ({generation_time/total_time*100:.1f}%)")
        print(f"   文件导出: {export_time:.2f}秒 ({export_time/total_time*100:.1f}%)")
        print(f"   处理速度: {(args.size**3) / generation_time:,.0f} 点/秒")

        print(f"\n📁 成功导出 {len(exported_files)} 个文件:")

        total_size = 0
        for file_path in exported_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                abs_path = os.path.abspath(file_path)
                total_size += size
                print(f"  ✓ {abs_path} ({size:,} bytes)")

        print(f"\n💾 总文件大小: {total_size / 1024 / 1024:.1f} MB")

        print(f"\n🔧 使用方法:")
        print(f"   将生成的 .cube 文件导入支持3D LUT的软件:")

        print(f"\n📹 视频编辑软件:")
        print(f"   • Adobe Premiere Pro: 效果面板 → Lumetri颜色 → 输入LUT")
        print(f"   • DaVinci Resolve: 色彩页面 → LUTs → 添加3D LUT")
        print(f"   • Final Cut Pro: 效果浏览器 → 颜色 → 颜色查找表 → 3D LUT")
        print(f"   • After Effects: 效果 → 颜色校正 → 应用颜色LUT")

        print(f"\n🖼️ 静态图像软件:")
        print(f"   • Adobe Photoshop: 图层 → 新建调整图层 → 颜色查找 → 3DLUT文件")
        print(f"   • Affinity Photo: 调整图层 → 重新映射 → 3D LUT")

        print(f"\n📷 专业摄影软件:")
        print(f"   • Adobe Lightroom 和 Adobe Camera Raw:")
        print(f"     1. 打开 Photoshop，随便打开一张 Raw 格式图片（或使用 滤镜 > Camera Raw 滤镜）")
        print(f"     2. **关键步骤**: 打开预设面板, 按住键盘上的 Alt 键 (Windows) 或 Option 键 (Mac) 不松手, 同时点击\"新建预设\"图标")
        print(f"     3. 在弹出的\"新建配置文件 (New Profile)\"窗口中：")
        print(f"        - 勾选最下方的\"颜色查找表 (Color Lookup Table)\"")
        print(f"        - 点击文件夹图标，选择你的.cube文件（此时是可以选中的）")
        print(f"        - 起个名字，点击确定")
        print(f"     ")
        print(f"     使用方法:")
        print(f"     - 创建的配置文件会自动同步到Lightroom和Camera Raw")
        print(f"     - 在Lightroom的\"配置文件\"面板中可以找到新建的配置文件")
        print(f"     - 在Camera Raw的\"配置文件\"浏览器中可以使用")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断操作")
        return 1

    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())