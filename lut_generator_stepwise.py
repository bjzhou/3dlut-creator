import numpy as np
from typing import Tuple, Dict, List, Optional
from image_processor import ImageColorMapper
import time


import torch
TORCH_AVAILABLE = True

class LUT3DGeneratorStepwise:
    """Stepwise 3D LUT generator - processes images one by one to avoid memory issues"""

    def __init__(self, lut_size: int = 64, device: str = 'auto'):
        """
        Initialize stepwise 3D LUT generator

        Args:
            lut_size: LUT grid size, default 64 (64x64x64)
            device: Device to use ('cpu', 'mps', 'cuda', 'auto')
        """
        self.lut_size = lut_size
        self.lut_data: Optional[np.ndarray] = None

        # Determine device
        if device == 'auto':
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                    print("Using Metal Performance Shaders (MPS) for acceleration")
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                    print("Using CUDA for acceleration")
                else:
                    self.device = 'cpu'
                    print("GPU not available, using CPU")
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.torch_available = TORCH_AVAILABLE and self.device != 'cpu'

    def process_image_pair(self, photoa_path: str, photob_path: str) -> np.ndarray:
        """
        Process a single image pair and return partial LUT contribution

        Args:
            photoa_path: Base image path
            photob_path: Mapped image path

        Returns:
            Partial color mapping contribution as numpy array
        """
        print(f"处理图片对: {photoa_path.split('/')[-1]}")

        # Load images
        from PIL import Image
        try:
            img_a = Image.open(photoa_path)
            img_b = Image.open(photob_path)

            # Convert to RGB if needed
            if img_a.mode != 'RGB':
                img_a = img_a.convert('RGB')
            if img_b.mode != 'RGB':
                img_b = img_b.convert('RGB')

            rgb_a = np.array(img_a)
            rgb_b = np.array(img_b)

            if rgb_a.shape != rgb_b.shape:
                print(f"警告: 图片尺寸不一致，跳过 {photoa_path}")
                return np.array([])

        except Exception as e:
            print(f"读取图片失败 {photoa_path}: {e}")
            return np.array([])

        # Extract unique color mappings from this image
        height, width = rgb_a.shape[:2]
        unique_mappings = {}

        print(f"  提取颜色映射 ({width}x{height} = {width*height:,} 像素)...")

        # Process pixels in chunks to avoid memory issues
        chunk_size = 10000  # Process 10k pixels at a time
        total_pixels = height * width

        for start_idx in range(0, total_pixels, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pixels)

            # Convert flat index to 2D coordinates
            for flat_idx in range(start_idx, end_idx):
                y = flat_idx // width
                x = flat_idx % width

                rgb_in = tuple(rgb_a[y, x])
                rgb_out = tuple(rgb_b[y, x])

                if rgb_in not in unique_mappings:
                    unique_mappings[rgb_in] = []
                unique_mappings[rgb_in].append(rgb_out)

            # Progress update
            if end_idx % 50000 == 0 or end_idx == total_pixels:
                progress = (end_idx / total_pixels) * 100
                print(f"  进度: {progress:.1f}%", end='\r')

        print()  # New line

        # Average multiple mappings for same input color
        averaged_mappings = {}
        for rgb_in, rgb_out_list in unique_mappings.items():
            if rgb_out_list:
                avg_rgb = tuple(int(round(np.mean(rgb_out_list, axis=0)[i])) for i in range(3))
                averaged_mappings[rgb_in] = avg_rgb

        print(f"  提取到 {len(averaged_mappings)} 个唯一颜色映射")

        return averaged_mappings

    def generate_lut_grid(self) -> np.ndarray:
        """
        Generate 3D LUT grid coordinates

        Returns:
            Grid points array with shape (lut_size^3, 3)
        """
        grid_points = []

        for b in range(self.lut_size):
            for g in range(self.lut_size):
                for r in range(self.lut_size):
                    # Convert grid coordinates to 0-255 range
                    r_val = int(r * 255.0 / (self.lut_size - 1))
                    g_val = int(g * 255.0 / (self.lut_size - 1))
                    b_val = int(b * 255.0 / (self.lut_size - 1))

                    grid_points.append([r_val, g_val, b_val])

        return np.array(grid_points, dtype=np.float32)

    def compress_color_mappings(self, color_mapping_dict: Dict[Tuple[int, int, int], Tuple[int, int, int]],
                              similarity_threshold: float = 3.0) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Compress color mappings by merging similar colors to reduce interpolation workload

        Args:
            color_mapping_dict: Original color mapping dictionary
            similarity_threshold: Color distance threshold for merging

        Returns:
            Compressed color mapping dictionary
        """
        print(f"压缩颜色映射，相似度阈值: {similarity_threshold}...")
        original_count = len(color_mapping_dict)

        if original_count < 10000:  # Don't compress small datasets
            print(f"数据集较小 ({original_count})，跳过压缩")
            return color_mapping_dict

        # Convert to numpy arrays for faster processing
        keys = np.array(list(color_mapping_dict.keys()), dtype=np.float32)
        values = np.array(list(color_mapping_dict.values()), dtype=np.float32)

        # Use spatial hashing for efficient clustering
        grid_size = int(similarity_threshold * 2)
        clusters = {}

        for i, (key, value) in enumerate(zip(keys, values)):
            # Find grid cell for this color
            grid_pos = tuple((key // grid_size).astype(int))

            if grid_pos not in clusters:
                clusters[grid_pos] = []
            clusters[grid_pos].append((key, value))

        # Merge similar colors within each cluster
        compressed_mappings = {}
        total_merged = 0

        for grid_pos, cluster_points in clusters.items():
            if len(cluster_points) == 1:
                key_tuple = tuple(cluster_points[0][0].astype(int))
                value_tuple = tuple(cluster_points[0][1].astype(int))
                compressed_mappings[key_tuple] = value_tuple
            else:
                # Average similar colors
                cluster_keys = np.array([p[0] for p in cluster_points])
                cluster_values = np.array([p[1] for p in cluster_points])

                # Average the colors
                avg_key = np.mean(cluster_keys, axis=0)
                avg_value = np.mean(cluster_values, axis=0)

                key_tuple = tuple(avg_key.astype(int))
                value_tuple = tuple(np.round(avg_value).astype(int))
                compressed_mappings[key_tuple] = value_tuple
                total_merged += len(cluster_points) - 1

        compression_ratio = (1 - len(compressed_mappings) / original_count) * 100
        print(f"压缩完成: {original_count:,} -> {len(compressed_mappings):,} 个映射点")
        print(f"压缩率: {compression_ratio:.1f}% (合并了 {total_merged:,} 个相似点)")

        return compressed_mappings

    def fast_interpolation_gpu(self, grid_points: np.ndarray,
                               color_mapping_dict: Dict[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray:
        """
        Fast GPU-based interpolation using PyTorch

        Args:
            grid_points: Grid points to interpolate (N, 3)
            color_mapping_dict: Color mapping dictionary

        Returns:
            Interpolated colors (N, 3)
        """
        if not TORCH_AVAILABLE:
            print("PyTorch不可用，回退到CPU模式")
            return self.fast_interpolation_cpu_fallback(grid_points, color_mapping_dict)

        print(f"开始GPU插值计算，{len(color_mapping_dict)} 个映射点...")

        device = torch.device(self.device)
        n_grid_points = grid_points.shape[0]

        # Prepare data for GPU
        mapping_keys = torch.tensor(list(color_mapping_dict.keys()), dtype=torch.float32, device=device)
        mapping_values = torch.tensor(list(color_mapping_dict.values()), dtype=torch.float32, device=device)
        query_points = torch.tensor(grid_points, dtype=torch.float32, device=device)

        memory_usage = mapping_keys.nbytes / 1024 / 1024
        print(f"GPU数据准备完成，内存使用: {memory_usage:.1f} MB")

        # Batch processing to avoid memory issues
        batch_size = 5000  # Process 5k points at a time
        total_batches = (n_grid_points + batch_size - 1) // batch_size

        result = torch.zeros((n_grid_points, 3), dtype=torch.float32, device=device)
        start_time = time.time()

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_grid_points)

            batch_points = query_points[start:end]

            # Calculate distances using PyTorch
            distances = torch.cdist(batch_points, mapping_keys, p=2)  # Shape: (batch, n_mappings)

            # Find nearest neighbors
            k = min(16, len(mapping_keys))  # Use more neighbors for better quality
            nearest_distances, nearest_indices = torch.topk(distances, k, largest=False, dim=1)

            # Create result tensor
            batch_result = torch.zeros((end - start, 3), dtype=torch.float32, device=device)

            for i in range(end - start):
                point_distances = nearest_distances[i]
                point_indices = nearest_indices[i]
                point_values = mapping_values[point_indices]

                # Check for exact match
                if torch.min(point_distances) < 1.0:
                    closest_idx = torch.argmin(point_distances)
                    batch_result[i] = point_values[closest_idx]
                else:
                    # Inverse distance weighting
                    weights = 1.0 / (point_distances + 1e-6)
                    weights = weights / torch.sum(weights)
                    batch_result[i] = torch.sum(weights.unsqueeze(1) * point_values, dim=0)

            result[start:end] = batch_result

            # Progress update
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                progress = ((batch_idx + 1) / total_batches) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                print(f"GPU插值进度: {progress:.1f}% (预计剩余 {eta:.0f}秒)", end='\r')

        print()  # New line
        print(f"GPU插值完成，耗时: {time.time() - start_time:.1f}秒")

        # Move back to CPU
        return result.cpu().numpy()

    def fast_interpolation_cpu_fallback(self, grid_points: np.ndarray,
                                       color_mapping_dict: Dict[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray:
        """
        Optimized CPU interpolation with spatial indexing

        Args:
            grid_points: Grid points to interpolate (N, 3)
            color_mapping_dict: Color mapping dictionary

        Returns:
            Interpolated colors (N, 3)
        """
        print(f"使用优化CPU插值，{len(color_mapping_dict)} 个映射点...")

        # Create spatial index for faster lookup
        grid_size = 16  # Create 16x16x16 spatial grid
        spatial_index = {}

        for rgb_in, rgb_out in color_mapping_dict.items():
            grid_pos = (rgb_in[0] // grid_size, rgb_in[1] // grid_size, rgb_in[2] // grid_size)
            if grid_pos not in spatial_index:
                spatial_index[grid_pos] = []
            spatial_index[grid_pos].append((rgb_in, rgb_out))

        n_grid_points = grid_points.shape[0]
        result = np.zeros((n_grid_points, 3), dtype=np.float32)

        start_time = time.time()
        batch_size = 2000  # Larger batch size for CPU
        total_batches = (n_grid_points + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_grid_points)

            for i in range(start, end):
                point = grid_points[i]
                point_int = tuple(point.astype(int))

                # Check for exact match first
                if point_int in color_mapping_dict:
                    result[i] = color_mapping_dict[point_int]
                    continue

                # Search nearby spatial cells
                grid_pos = (point[0] // grid_size, point[1] // grid_size, point[2] // grid_size)
                candidates = []

                # Search current cell and neighboring cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            search_pos = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                            if search_pos in spatial_index:
                                candidates.extend(spatial_index[search_pos])

                if not candidates:
                    # Fallback to global search
                    candidates = list(color_mapping_dict.items())

                # Find nearest neighbors
                if len(candidates) > 0:
                    candidate_keys = np.array([c[0] for c in candidates], dtype=np.float32)
                    candidate_values = np.array([c[1] for c in candidates], dtype=np.float32)

                    distances = np.linalg.norm(candidate_keys - point, axis=1)
                    k = min(8, len(candidates))
                    # 确保k不会超出数组边界 (k-1用于argpartition)
                    k = min(k, len(distances))
                    if k == 0:
                        continue  # 跳过无效情况

                    nearest_indices = np.argpartition(distances, k-1)[:k]
                    nearest_values = candidate_values[nearest_indices]

                    # 获取对应的最小距离
                    nearest_distances = distances[nearest_indices]

                    if np.min(nearest_distances) < 2.0:
                        # 找到实际的最小值索引
                        actual_min_idx = np.argmin(nearest_distances)
                        result[i] = nearest_values[actual_min_idx]
                    else:
                        # Weighted average
                        weights = 1.0 / (nearest_distances + 1e-6)
                        weights = weights / np.sum(weights)
                        result[i] = np.sum(weights[:, np.newaxis] * nearest_values, axis=0)

            # Progress update
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                progress = ((batch_idx + 1) / total_batches) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                print(f"优化CPU插值进度: {progress:.1f}% (预计剩余 {eta:.0f}秒)", end='\r')

        print()  # New line
        print(f"优化CPU插值完成，耗时: {time.time() - start_time:.1f}秒")

        return result

    def generate_3d_lut_stepwise(self, photoa_dir: str, photob_dir: str) -> np.ndarray:
        """
        Generate 3D LUT by processing images step by step

        Args:
            photoa_dir: Base images directory
            photob_dir: Mapped images directory

        Returns:
            3D LUT data array with shape (lut_size, lut_size, lut_size, 3)
        """
        print(f"开始分步生成 {self.lut_size}x{self.lut_size}x{self.lut_size} 的3D LUT...")
        print(f"使用设备: {self.device.upper()}")

        # Find all image pairs
        import glob
        import os

        photoa_files = glob.glob(os.path.join(photoa_dir, "*"))
        photoa_files = [f for f in photoa_files if os.path.isfile(f) and
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        image_pairs = []
        for photoa_path in photoa_files:
            filename = os.path.basename(photoa_path)
            photob_path = os.path.join(photob_dir, filename)

            if os.path.exists(photob_path):
                image_pairs.append((photoa_path, photob_path))

        print(f"找到 {len(image_pairs)} 对图片")

        # Process images step by step
        start_time = time.time()
        all_color_mappings = {}

        for i, (photoa_path, photob_path) in enumerate(image_pairs):
            print(f"\n--- 处理图片 {i+1}/{len(image_pairs)} ---")

            # Process single image pair
            image_mappings = self.process_image_pair(photoa_path, photob_path)

            # Direct merge - since process_image_pair already averaged the mappings
            for rgb_in, rgb_out in image_mappings.items():
                if rgb_in not in all_color_mappings:
                    # Store as list for potential future merging
                    all_color_mappings[rgb_in] = [rgb_out]
                else:
                    # Average with existing mappings
                    existing_value = all_color_mappings[rgb_in][0]  # First (and only) value
                    # Average the two mapping values
                    avg_rgb = tuple(int(round((existing_value[j] + rgb_out[j]) / 2.0)) for j in range(3))
                    all_color_mappings[rgb_in][0] = avg_rgb

            print(f"  当前总映射数: {len(all_color_mappings)}")

            # Periodic memory optimization
            if len(all_color_mappings) > 200000 and i % 3 == 0:  # Every 3 images
                print(f"内存优化: {len(all_color_mappings)} 个映射点，保持高效处理")
                # all_color_mappings已经保持了精简状态，无需额外优化

        print(f"\n--- 最终处理完成 ---")
        final_mappings = {}
        for rgb_in, rgb_out_list in all_color_mappings.items():
            if rgb_out_list and len(rgb_out_list) > 0:
                # Use the first (and only) value since we keep it averaged
                final_mappings[rgb_in] = rgb_out_list[0]

        processing_time = time.time() - start_time
        print(f"图片处理完成，耗时: {processing_time:.1f}秒")
        print(f"最终颜色映射数量: {len(final_mappings)}")

        # Generate grid points
        print(f"\n--- 生成3D LUT网格 ---")
        grid_points = self.generate_lut_grid()
        total_points = len(grid_points)
        print(f"需要计算 {total_points:,} 个网格点")

        # Perform interpolation
        print(f"\n--- 插值计算 ---")
        interpolation_start = time.time()

        # Compress color mappings to reduce computation
        compressed_mappings = self.compress_color_mappings(final_mappings, similarity_threshold=2)

        # Use GPU acceleration if available
        if self.torch_available:
            mapped_colors = self.fast_interpolation_gpu(grid_points, compressed_mappings)
        else:
            mapped_colors = self.fast_interpolation_cpu_fallback(grid_points, compressed_mappings)

        interpolation_time = time.time() - interpolation_start
        print(f"插值计算完成，耗时: {interpolation_time:.1f}秒")

        # Convert to 0-1 range
        mapped_colors_norm = mapped_colors / 255.0
        mapped_colors_norm = np.clip(mapped_colors_norm, 0.0, 1.0)

        # Reshape to 3D LUT format
        lut_data_3d = mapped_colors_norm.reshape(self.lut_size, self.lut_size, self.lut_size, 3)

        total_time = time.time() - start_time
        print(f"\n✅ 3D LUT生成完成!")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"处理速度: {(total_points / total_time):,.0f} 点/秒")

        self.lut_data = lut_data_3d
        return lut_data_3d


def test_stepwise_generator():
    """Test the stepwise LUT generator"""
    print("测试分步LUT生成器...")

    # Create test directories and images
    import os
    from PIL import Image, ImageDraw

    os.makedirs("test_photoa", exist_ok=True)
    os.makedirs("test_photb", exist_ok=True)

    # Create a few test images
    for i in range(5):  # 5 pairs
        # Base image (various colors)
        img_a = Image.new('RGB', (200, 150))
        draw_a = ImageDraw.Draw(img_a)

        # Draw different color patterns
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for x in range(200):
            for y in range(150):
                color_idx = (x + y + i * 50) % len(colors)
                draw_a.point((x, y), colors[color_idx])

        img_a.save(f"test_photoa/test_image_{i}.png")

        # Mapped image (color shifted)
        img_b = Image.new('RGB', (200, 150))
        draw_b = ImageDraw.Draw(img_b)

        for x in range(200):
            for y in range(150):
                r, g, b = draw_a.getpixel((x, y))
                # Apply color transformation
                new_r = min(255, int(r * 1.2))
                new_g = min(255, int(g * 0.8))
                new_b = min(255, int(b * 1.1))
                draw_b.point((x, y), (new_r, new_g, new_b))

        img_b.save(f"test_photb/test_image_{i}.png")

    # Test the generator
    print("测试分步生成器...")
    generator = LUT3DGeneratorStepwise(lut_size=32, device='cpu')

    try:
        import time
        start_time = time.time()

        lut_data = generator.generate_3d_lut_stepwise("test_photoa", "test_photb")

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\n✅ 测试成功!")
        print(f"LUT数据形状: {lut_data.shape}")
        print(f"生成时间: {elapsed:.2f}秒")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_stepwise_generator()