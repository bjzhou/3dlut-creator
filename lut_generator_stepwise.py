import numpy as np
from typing import Tuple, Dict, List, Optional
from image_processor import ImageColorMapper
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os


import torch
TORCH_AVAILABLE = True


class GPUColorMappings:
    """
    GPU-native color mappings using tensors instead of dict.
    Keeps all data on GPU to eliminate CPU-GPU transfers.
    """
    
    def __init__(self, device='cuda', bit_depth=8):
        """
        Initialize GPU color mappings
        
        Args:
            device: GPU device ('cuda' or 'mps')
            bit_depth: Bit depth of the input images (default: 8)
        """
        self.device = torch.device(device)
        self.bit_depth = bit_depth
        self.max_val = (2 ** bit_depth) - 1
        self.multiplier = 2 ** bit_depth
        
        self.keys_tensor = None      # (N, 3) RGB input colors on GPU
        self.values_tensor = None    # (N, 3) RGB output colors on GPU
        self.weights_tensor = None   # (N,) Weights/counts for each mapping

        
    def add_batch(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Add a batch of mappings (already on GPU)
        
        Args:
            new_keys: (M, 3) tensor of input RGB colors
            new_values: (M, 3) tensor of output RGB colors
        """
        if self.keys_tensor is None:
            self.keys_tensor = new_keys
            self.values_tensor = new_values
            self.weights_tensor = torch.ones(len(new_keys), dtype=torch.float32, device=self.device)
        else:
            # Concatenate tensors (GPU operation, no CPU transfer)
            self.keys_tensor = torch.cat([self.keys_tensor, new_keys], dim=0)
            self.values_tensor = torch.cat([self.values_tensor, new_values], dim=0)
            
            # Handle weights
            new_weights = torch.ones(len(new_keys), dtype=torch.float32, device=self.device)
            if self.weights_tensor is None:
                self.weights_tensor = torch.ones(len(self.keys_tensor) - len(new_keys), 
                                               dtype=torch.float32, device=self.device)
            self.weights_tensor = torch.cat([self.weights_tensor, new_weights], dim=0)

    
    def unique_and_merge(self):
        """
        Remove duplicate keys and average their values (all on GPU)
        """
        if self.keys_tensor is None:
            return
        
        # Encode RGB to single integer key for uniqueness
        # Use multipliers based on bit depth to avoid collisions
        m1 = 1
        m2 = self.multiplier
        m3 = self.multiplier * self.multiplier
        
        encoded = (self.keys_tensor[:, 0].long() * m1 + 
                   self.keys_tensor[:, 1].long() * m2 + 
                   self.keys_tensor[:, 2].long() * m3)
        
        # Initialize weights if needed
        if self.weights_tensor is None:
            self.weights_tensor = torch.ones(len(self.keys_tensor), dtype=torch.float32, device=self.device)

        # Find unique keys (GPU operation)
        unique_encoded, inverse_indices = torch.unique(encoded, return_inverse=True)
        n_unique = len(unique_encoded)
        
        # Aggregate weights
        merged_weights = torch.zeros(n_unique, dtype=torch.float32, device=self.device)
        merged_weights.scatter_add_(0, inverse_indices, self.weights_tensor)
        
        # Calculate weighted sum of values
        merged_values = torch.zeros(n_unique, 3, dtype=torch.float32, device=self.device)
        # Weighted values
        weighted_values = self.values_tensor * self.weights_tensor.unsqueeze(1)
        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        merged_values.scatter_add_(0, indices_expanded, weighted_values)
        
        # Calculate averages
        merged_values = merged_values / merged_weights.unsqueeze(1)
        
        # Decode unique keys back to RGB
        m1 = 1
        m2 = self.multiplier
        m3 = self.multiplier * self.multiplier
        
        merged_keys = torch.zeros(n_unique, 3, dtype=torch.float32, device=self.device)
        merged_keys[:, 0] = unique_encoded % m2
        merged_keys[:, 1] = (unique_encoded // m2) % m2
        merged_keys[:, 2] = unique_encoded // m3
        
        self.keys_tensor = merged_keys
        self.values_tensor = merged_values
        self.weights_tensor = merged_weights
    
    def compress_spatial(self, threshold: float = 3.0):
        """
        Compress mappings by merging spatially close colors (all on GPU)
        
        Args:
            threshold: Distance threshold for merging
        """
        if self.keys_tensor is None or len(self.keys_tensor) < 10000:
            return
        
        grid_size = int(threshold * 2)
        
        # Initialize weights if needed
        if self.weights_tensor is None:
            self.weights_tensor = torch.ones(len(self.keys_tensor), dtype=torch.float32, device=self.device)
        
        # Calculate grid indices
        grid_indices = (self.keys_tensor / grid_size).long()
        
        # Encode grid position as single key
        # Use a large enough multiplier to avoid overlap in 16-bit mode
        m1 = 1
        m2 = self.multiplier
        m3 = self.multiplier * self.multiplier
        
        grid_keys = (grid_indices[:, 0] * m1 + 
                     grid_indices[:, 1] * m2 + 
                     grid_indices[:, 2] * m3)
        
        # Find unique grids
        unique_grids, inverse_indices = torch.unique(grid_keys, return_inverse=True)
        n_grids = len(unique_grids)
        
        # Compress weights
        compressed_weights = torch.zeros(n_grids, dtype=torch.float32, device=self.device)
        compressed_weights.scatter_add_(0, inverse_indices, self.weights_tensor)
        
        # Aggregate weighted keys and values
        compressed_keys = torch.zeros(n_grids, 3, dtype=torch.float32, device=self.device)
        compressed_values = torch.zeros(n_grids, 3, dtype=torch.float32, device=self.device)
        
        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        
        # Weighted accumulation
        weighted_keys = self.keys_tensor * self.weights_tensor.unsqueeze(1)
        weighted_values = self.values_tensor * self.weights_tensor.unsqueeze(1)
        
        compressed_keys.scatter_add_(0, indices_expanded, weighted_keys)
        compressed_values.scatter_add_(0, indices_expanded, weighted_values)
        
        # Calculate weighted averages
        compressed_keys = compressed_keys / compressed_weights.unsqueeze(1)
        compressed_values = compressed_values / compressed_weights.unsqueeze(1)
        
        self.keys_tensor = compressed_keys
        self.values_tensor = compressed_values
        self.weights_tensor = compressed_weights
    
    def to_dict(self) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Convert to dict (downloads from GPU to CPU - use only when needed)
        
        Returns:
            Dictionary of color mappings
        """
        if self.keys_tensor is None:
            return {}
        
        # Download to CPU
        keys_cpu = self.keys_tensor.cpu().numpy()
        values_cpu = self.values_tensor.cpu().numpy()
        
        # Build dict
        result = {}
        # Avoid rounding issues for 16-bit by using float-aware key if needed, 
        # but here we stick to int for dictionary keys as expected.
        for i in range(len(keys_cpu)):
            key_tuple = tuple(np.round(keys_cpu[i]).astype(int))
            value_tuple = tuple(np.round(values_cpu[i]).astype(int))
            result[key_tuple] = value_tuple
        
        return result
    
    def size(self) -> int:
        """Return number of mappings"""
        return 0 if self.keys_tensor is None else len(self.keys_tensor)
    
    def rgb_to_oklab_tensor(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to Oklab color space on GPU.
        Oklab is more perceptually uniform than CIELAB and has better hue constancy.
        
        Args:
            rgb_tensor: RGB tensor in range [0, max_val], shape (..., 3)
        
        Returns:
            Oklab tensor: L in [0, 1], a and b in approx [-0.4, 0.4]
        """
        # Normalize RGB to [0, 1]
        rgb = rgb_tensor / float(self.max_val)
        
        # Convert to linear RGB (sRGB inverse gamma)
        rgb_linear = torch.where(
            rgb > 0.04045,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92
        )
        
        # Linear RGB to LMS
        l = 0.4122214708 * rgb_linear[..., 0] + 0.5363325363 * rgb_linear[..., 1] + 0.0514459929 * rgb_linear[..., 2]
        m = 0.2119034982 * rgb_linear[..., 0] + 0.6806995451 * rgb_linear[..., 1] + 0.1073969566 * rgb_linear[..., 2]
        s = 0.0883024619 * rgb_linear[..., 0] + 0.2817188976 * rgb_linear[..., 1] + 0.6299787005 * rgb_linear[..., 2]
        
        # Non-linearity
        l_ = torch.pow(l.clamp(min=0), 1.0/3.0)
        m_ = torch.pow(m.clamp(min=0), 1.0/3.0)
        s_ = torch.pow(s.clamp(min=0), 1.0/3.0)
        
        # LMS to Oklab
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        
        return torch.stack([L, a, b], dim=-1)
    
    def filter_by_delta_e(self, max_delta_e: float = 50.0, percentile_threshold: float = 95.0):
        """
        Filter out mappings with excessive color difference (Delta E)
        This helps remove outliers caused by noise, overexposure, or alignment issues.
        
        Args:
            max_delta_e: Maximum allowed Delta E. Mappings exceeding this are removed.
                        Typical values:
                        - 2.3: Just noticeable difference
                        - 10-20: Normal color grading range
                        - 30-50: Significant color shift (used for LUT generation)
                        - >50: Likely outliers/errors
            percentile_threshold: Only keep mappings with Delta E below this percentile.
                                 e.g., 95 means remove the top 5% most extreme mappings.
        """
        if self.keys_tensor is None or len(self.keys_tensor) == 0:
            return
        
        # Scale max_delta_e for Oklab (original was 50.0 for LAB, Oklab range is ~1.0)
        # Delta E in Oklab is much smaller. ~0.5 Oklab distance is a large shift.
        max_delta_e_ok = max_delta_e / 100.0
        
        print(f"    过滤异常映射 (Oklab Delta E 阈值: {max_delta_e_ok:.3f}, 百分位: {percentile_threshold}%)...")
        original_size = len(self.keys_tensor)
        
        # Convert to Oklab for perceptually uniform distance
        keys_ok = self.rgb_to_oklab_tensor(self.keys_tensor)
        values_ok = self.rgb_to_oklab_tensor(self.values_tensor)
        
        # Calculate Delta E (Euclidean distance in Oklab space)
        delta_e = torch.sqrt(torch.sum((keys_ok - values_ok) ** 2, dim=1))
        
        # Create mask for valid mappings
        # Condition 1: Delta E below max threshold
        mask_max = delta_e <= max_delta_e_ok
        
        # Condition 2: Delta E below percentile threshold
        percentile_value = torch.quantile(delta_e, percentile_threshold / 100.0)
        mask_percentile = delta_e <= percentile_value
        
        # Combine both conditions
        valid_mask = mask_max & mask_percentile
        
        # Apply filter
        self.keys_tensor = self.keys_tensor[valid_mask]
        self.values_tensor = self.values_tensor[valid_mask]
        if self.weights_tensor is not None:
            self.weights_tensor = self.weights_tensor[valid_mask]
        
        filtered_count = original_size - len(self.keys_tensor)
        print(f"    剔除异常点: {filtered_count:,} 个 ({filtered_count/original_size*100:.1f}%)")
        print(f"    Delta E 统计: 中位数={torch.median(delta_e[valid_mask]):.1f}, "
              f"最大={delta_e[valid_mask].max():.1f}, P95={percentile_value:.1f}")

    def filter_outliers_by_local_consistency(self, grid_size: int = 8, std_threshold: float = 2.5):
        """
        Filter outliers based on local color consistency using spatial grid grouping.
        Uses O(N) memory instead of O(N²) by grouping colors into spatial cells.
        
        For each mapping, check if its output color is consistent with other mappings
        in the same spatial cell of color space.
        
        Args:
            grid_size: Size of grid cells in Oklab space for local grouping (scaled internally)
            std_threshold: Remove mappings whose output deviates more than this many 
                          standard deviations from cell average
        """
        if self.keys_tensor is None or len(self.keys_tensor) < 100:
            return
        
        # Adjust grid_size for Oklab (L: [0,1], a,b: ~[-0.4, 0.4])
        # A grid size of 8 in LAB (L range 100) is equivalent to 0.08 in Oklab
        grid_size_ok = grid_size / 100.0
        
        print(f"    局部一致性过滤 (Oklab网格={grid_size_ok:.3f}, std阈值={std_threshold})...")
        original_size = len(self.keys_tensor)
        
        # Convert to Oklab for perceptually uniform analysis
        keys_ok = self.rgb_to_oklab_tensor(self.keys_tensor)
        values_ok = self.rgb_to_oklab_tensor(self.values_tensor)
        
        # Quantize Oklab to grid cells
        # L in [0, 1], a and b approx in [-0.4, 0.4]
        keys_grid = torch.zeros_like(keys_ok, dtype=torch.long)
        keys_grid[:, 0] = (keys_ok[:, 0] / grid_size_ok).long().clamp(0, 255)  # L
        keys_grid[:, 1] = ((keys_ok[:, 1] + 0.4) / grid_size_ok).long().clamp(0, 255)  # a
        keys_grid[:, 2] = ((keys_ok[:, 2] + 0.4) / grid_size_ok).long().clamp(0, 255)  # b
        
        # Encode grid cell as single integer
        cell_ids = keys_grid[:, 0] + keys_grid[:, 1] * 256 + keys_grid[:, 2] * 65536
        
        # Find unique cells and group mappings
        unique_cells, inverse_indices = torch.unique(cell_ids, return_inverse=True)
        n_cells = len(unique_cells)
        
        # Calculate per-cell statistics using scatter operations (memory efficient)
        # Sum of values per cell
        cell_sum = torch.zeros(n_cells, 3, dtype=torch.float32, device=self.device)
        cell_sum_sq = torch.zeros(n_cells, 3, dtype=torch.float32, device=self.device)
        cell_count = torch.zeros(n_cells, dtype=torch.float32, device=self.device)
        
        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        cell_sum.scatter_add_(0, indices_expanded, values_ok)
        cell_sum_sq.scatter_add_(0, indices_expanded, values_ok ** 2)
        cell_count.scatter_add_(0, inverse_indices, torch.ones(len(values_ok), device=self.device))
        
        # Calculate cell mean and std
        cell_mean = cell_sum / cell_count.unsqueeze(1).clamp(min=1)
        cell_var = (cell_sum_sq / cell_count.unsqueeze(1).clamp(min=1)) - (cell_mean ** 2)
        cell_std = torch.sqrt(cell_var.clamp(min=0))
        
        # Get statistics for each point's cell
        point_cell_mean = cell_mean[inverse_indices]  # (N, 3)
        point_cell_std = cell_std[inverse_indices]  # (N, 3)
        point_cell_count = cell_count[inverse_indices]  # (N,)
        
        # Calculate deviation from cell mean
        deviation = torch.abs(values_ok - point_cell_mean)
        
        # Normalize by std (with minimum to avoid division issues)
        # Adjust min_std for Oklab range
        min_std_ok = 0.02 
        normalized_deviation = deviation / (point_cell_std + min_std_ok)
        max_deviation = normalized_deviation.max(dim=1).values
        
        # Mark as valid if:
        # 1. Within std threshold, OR
        # 2. Cell has very few samples (keep them for now)
        valid_mask = (max_deviation <= std_threshold) | (point_cell_count < 3)
        
        # Apply filter
        self.keys_tensor = self.keys_tensor[valid_mask]
        self.values_tensor = self.values_tensor[valid_mask]
        if self.weights_tensor is not None:
            self.weights_tensor = self.weights_tensor[valid_mask]
        
        # Clean up
        del keys_ok, values_ok, keys_grid, cell_ids, cell_sum, cell_sum_sq
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        
        filtered_count = original_size - len(self.keys_tensor)
        print(f"    局部异常点剔除: {filtered_count:,} 个 ({filtered_count/original_size*100:.1f}%)")

    def remove_extreme_colors(self, margin: int = 5):
        """
        Remove mappings at extreme edges of color space (near 0 or 255).
        These often have unreliable mappings due to clipping/saturation.
        
        Args:
            margin: Remove colors within this distance from 0 or 255
        """
        if self.keys_tensor is None:
            return
        
        original_size = len(self.keys_tensor)
        
        # Find colors too close to extremes
        too_dark = (self.keys_tensor < margin).any(dim=1)
        too_bright = (self.keys_tensor > self.max_val - margin).any(dim=1)
        extreme_mask = too_dark | too_bright
        
        # Keep non-extreme colors
        valid_mask = ~extreme_mask
        self.keys_tensor = self.keys_tensor[valid_mask]
        self.values_tensor = self.values_tensor[valid_mask]
        if self.weights_tensor is not None:
            self.weights_tensor = self.weights_tensor[valid_mask]
        
        filtered_count = original_size - len(self.keys_tensor)
        if filtered_count > 0:
            print(f"    移除极端色彩: {filtered_count:,} 个 ({filtered_count/original_size*100:.1f}%)")
    
    def clear_memory(self):
        """Release GPU memory"""
        del self.keys_tensor, self.values_tensor
        self.keys_tensor = None
        self.values_tensor = None
        self.weights_tensor = None
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()


class LUT3DGeneratorStepwise:
    """Stepwise 3D LUT generator - processes images one by one to avoid memory issues"""

    def __init__(self, lut_size: int = 64, device: str = 'auto', bit_depth: int = 8):
        """
        Initialize stepwise 3D LUT generator

        Args:
            lut_size: LUT grid size, default 64 (64x64x64)
            device: Device to use ('cpu', 'mps', 'cuda', 'auto')
            bit_depth: Bit depth of the input images (default: 8)
        """
        self.lut_size = lut_size
        self.lut_data: Optional[np.ndarray] = None
        self.bit_depth = bit_depth
        self.max_val = (2 ** bit_depth) - 1

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
        
        # Enable GPU acceleration for pixel collection (can be disabled if needed)
        self.use_gpu_for_pixel_collection = self.torch_available
        if self.use_gpu_for_pixel_collection:
            print(f"GPU acceleration enabled for pixel collection on {self.device.upper()}")

    def process_image_pair_gpu_native(self, photoa_path: str, photob_path: str, 
                                        gpu_mappings: GPUColorMappings):
        """
        Process image pair and add directly to GPU tensor (zero-copy)
        
        Args:
            photoa_path: Base image path
            photob_path: Mapped image path
            gpu_mappings: GPUColorMappings object to add results to
        """
        filename = os.path.basename(photoa_path)
        
        # Load images
        from PIL import Image
        try:
            img_a = Image.open(photoa_path)
            img_b = Image.open(photob_path)

            # Check for 16-bit
            is_16bit_a = '16' in img_a.mode or img_a.mode == 'I'
            is_16bit_b = '16' in img_b.mode or img_b.mode == 'I'
            
            if is_16bit_a:
                rgb_a = np.array(img_a, dtype=np.uint16)
            else:
                if img_a.mode != 'RGB':
                    img_a = img_a.convert('RGB')
                rgb_a = np.array(img_a, dtype=np.uint8)
                
            if is_16bit_b:
                rgb_b = np.array(img_b, dtype=np.uint16)
            else:
                if img_b.mode != 'RGB':
                    img_b = img_b.convert('RGB')
                rgb_b = np.array(img_b, dtype=np.uint8)

            # Scale to match target bit depth
            # This ensures that even if inputs are 8-bit, they are treated 
            # as the correct range (e.g., 0-65535) if bit_depth=16 is selected.
            if self.bit_depth == 16:
                if not is_16bit_a:
                    rgb_a = rgb_a.astype(np.uint16) * 257
                if not is_16bit_b:
                    rgb_b = rgb_b.astype(np.uint16) * 257
            elif self.bit_depth == 8:
                if is_16bit_a:
                    rgb_a = (rgb_a.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
                if is_16bit_b:
                    rgb_b = (rgb_b.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)

            if rgb_a.shape != rgb_b.shape:
                print(f"  ⚠ 跳过 (尺寸不匹配): {filename}")
                return

        except Exception as e:
            print(f"  ⚠ 失败: {filename} - {e}")
            return

        # Process on GPU
        start_time = time.time()
        device = torch.device(self.device)
        
        pixels_a = torch.from_numpy(rgb_a.reshape(-1, 3)).to(device)
        pixels_b = torch.from_numpy(rgb_b.reshape(-1, 3)).to(device)

        # Encode & unique (GPU)
        # Use multipliers based on bit depth
        m2 = gpu_mappings.multiplier
        m3 = m2 * m2
        
        keys_a = (pixels_a[:, 0].long() + 
                  pixels_a[:, 1].long() * m2 + 
                  pixels_a[:, 2].long() * m3)

        unique_keys, inverse_indices = torch.unique(keys_a, return_inverse=True)
        counts = torch.bincount(inverse_indices, minlength=len(unique_keys))

        sum_rgb = torch.zeros(len(unique_keys), 3, dtype=torch.float32, device=device)
        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        sum_rgb.scatter_add_(0, indices_expanded, pixels_b.float())
        mean_rgb = sum_rgb / counts.unsqueeze(1).float()

        # Decode keys (GPU)
        unique_rgb_keys = torch.zeros(len(unique_keys), 3, dtype=torch.float32, device=device)
        unique_rgb_keys[:, 0] = unique_keys % m2
        unique_rgb_keys[:, 1] = (unique_keys // m2) % m2
        unique_rgb_keys[:, 2] = unique_keys // m3

        # Add to GPU mappings (NO CPU DOWNLOAD!)
        gpu_mappings.add_batch(unique_rgb_keys, mean_rgb)

        # Clean up
        del pixels_a, pixels_b, keys_a, unique_keys, inverse_indices, sum_rgb
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print(f"  ✓ {filename}: {len(unique_rgb_keys):,} colors ({elapsed*1000:.0f}ms)")

    def rgb_to_oklab_gpu(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to Oklab color space on GPU using PyTorch
        
        Args:
            rgb_tensor: RGB tensor in range [0, max_val], shape (..., 3), on GPU
        
        Returns:
            Oklab tensor: L in [0, 1], a and b in approx [-0.4, 0.4]
        """
        # Normalize RGB to [0, 1]
        rgb = rgb_tensor / float(self.max_val)
        
        # Convert to linear RGB (sRGB inverse gamma)
        rgb_linear = torch.where(
            rgb > 0.04045,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92
        )
        
        # Linear RGB to LMS
        l = 0.4122214708 * rgb_linear[..., 0] + 0.5363325363 * rgb_linear[..., 1] + 0.0514459929 * rgb_linear[..., 2]
        m = 0.2119034982 * rgb_linear[..., 0] + 0.6806995451 * rgb_linear[..., 1] + 0.1073969566 * rgb_linear[..., 2]
        s = 0.0883024619 * rgb_linear[..., 0] + 0.2817188976 * rgb_linear[..., 1] + 0.6299787005 * rgb_linear[..., 2]
        
        # Non-linearity
        l_ = torch.pow(l.clamp(min=0), 1.0/3.0)
        m_ = torch.pow(m.clamp(min=0), 1.0/3.0)
        s_ = torch.pow(s.clamp(min=0), 1.0/3.0)
        
        # LMS to Oklab
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        
        return torch.stack([L, a, b], dim=-1)

    def oklab_to_rgb_gpu(self, oklab_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert Oklab back to RGB color space on GPU using PyTorch
        
        Args:
            oklab_tensor: Oklab tensor: L in [0, 1], a and b in approx [-0.4, 0.4]
        
        Returns:
            RGB tensor in range [0, max_val], shape (..., 3)
        """
        L, a, b = oklab_tensor[..., 0], oklab_tensor[..., 1], oklab_tensor[..., 2]
        
        # Oklab to LMS
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        
        l = l_ ** 3
        m = m_ ** 3
        s = s_ ** 3
        
        # LMS to Linear RGB
        r_lin =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        
        rgb_linear = torch.stack([r_lin, g_lin, b_lin], dim=-1)
        
        # Linear RGB to sRGB (gamma)
        rgb = torch.where(
            rgb_linear > 0.0031308,
            1.055 * torch.pow(rgb_linear.clamp(min=0), 1.0 / 2.4) - 0.055,
            12.92 * rgb_linear
        )
        
        # Convert to [0, max_val] range and clip
        return torch.clamp(rgb * float(self.max_val), 0, self.max_val)

    @staticmethod
    def rgb_to_oklab(rgb: np.ndarray, max_val: float = 255.0) -> np.ndarray:
        """
        Convert RGB to Oklab color space (NumPy version)
        """
        rgb_norm = rgb / max_val
        rgb_linear = np.where(rgb_norm > 0.04045, np.power((rgb_norm + 0.055) / 1.055, 2.4), rgb_norm / 12.92)
        
        l = 0.4122214708 * rgb_linear[..., 0] + 0.5363325363 * rgb_linear[..., 1] + 0.0514459929 * rgb_linear[..., 2]
        m = 0.2119034982 * rgb_linear[..., 0] + 0.6806995451 * rgb_linear[..., 1] + 0.1073969566 * rgb_linear[..., 2]
        s = 0.0883024619 * rgb_linear[..., 0] + 0.2817188976 * rgb_linear[..., 1] + 0.6299787005 * rgb_linear[..., 2]
        
        l_ = np.power(np.maximum(0, l), 1.0/3.0)
        m_ = np.power(np.maximum(0, m), 1.0/3.0)
        s_ = np.power(np.maximum(0, s), 1.0/3.0)
        
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def oklab_to_rgb(oklab: np.ndarray, max_val: float = 255.0) -> np.ndarray:
        """
        Convert Oklab back to RGB color space (NumPy version)
        """
        L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
        
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        
        l, m, s = l_**3, m_**3, s_**3
        
        r_lin =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        
        rgb_linear = np.stack([r_lin, g_lin, b_lin], axis=-1)
        rgb = np.where(rgb_linear > 0.0031308, 1.055 * np.power(np.maximum(0, rgb_linear), 1.0 / 2.4) - 0.055, 12.92 * rgb_linear)
        
        return np.clip(rgb * max_val, 0, max_val)

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
                    # Convert grid coordinates to 0-max_val range
                    r_val = r * float(self.max_val) / (self.lut_size - 1)
                    g_val = g * float(self.max_val) / (self.lut_size - 1)
                    b_val = b * float(self.max_val) / (self.lut_size - 1)

                    grid_points.append([r_val, g_val, b_val])

        return np.array(grid_points, dtype=np.float32)

    def generate_3d_lut_stepwise(self, photoa_dir: str, photob_dir: str, num_threads: int = 4) -> np.ndarray:
        """
        Generate 3D LUT by processing images with multi-threading

        Args:
            photoa_dir: Base images directory
            photob_dir: Mapped images directory
            num_threads: Number of threads for parallel processing (default: 4)

        Returns:
            3D LUT data array with shape (lut_size, lut_size, lut_size, 3)
        """
        return self.generate_3d_lut_gpu_native(photoa_dir, photob_dir)

    def generate_3d_lut_gpu_native(self, photoa_dir: str, photob_dir: str) -> np.ndarray:
        """
        GPU-native LUT generation - all data stays on GPU until final output
        Eliminates all intermediate CPU-GPU transfers
        
        Args:
            photoa_dir: Directory containing base photos
            photob_dir: Directory containing mapped photos
            
        Returns:
            3D LUT data array
        """        
        print(f"\\n{'='*70}")
        print(f"GPU-Native LUT生成 (零CPU传输模式)")
        print(f"{'='*70}")
        print(f"开始生成 {self.lut_size}x{self.lut_size}x{self.lut_size} 的3D LUT...")
        print(f"使用设备: {self.device.upper()} (GPU-Native流程)")
        
        start_time = time.time()
        
        # Find image pairs
        photoa_files = sorted([os.path.join(photoa_dir, f) for f in os.listdir(photoa_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        
        image_pairs = []
        for photoa_path in photoa_files:
            filename = os.path.basename(photoa_path)
            photob_path = os.path.join(photob_dir, filename)
            if os.path.exists(photob_path):
                image_pairs.append((photoa_path, photob_path))
        
        print(f"找到 {len(image_pairs)} 对图片\\n")
        
        # Create GPU mappings object
        gpu_mappings = GPUColorMappings(self.device, bit_depth=self.bit_depth)
        
        # Process all images (data stays on GPU)
        print("阶段1: 像素收集 (GPU)")
        print("-" * 70)
        collection_start = time.time()
        
        for i, (photoa, photob) in enumerate(image_pairs, 1):
            print(f"[{i}/{len(image_pairs)}] ", end="")
            self.process_image_pair_gpu_native(photoa, photob, gpu_mappings)
        
        collection_time = time.time() - collection_start
        print(f"\\n像素收集完成: {collection_time:.1f}秒 ({collection_time/len(image_pairs):.2f}秒/张)")
        print(f"总映射数: {gpu_mappings.size():,}\\n")
        
        # Merge duplicates (GPU)
        print("阶段2: 合并重复 (GPU)")
        print("-" * 70)
        merge_start = time.time()
        gpu_mappings.unique_and_merge()
        merge_time = time.time() - merge_start
        print(f"合并完成: {gpu_mappings.size():,} 个唯一映射 ({merge_time:.2f}秒)\\n")
        
        # Filter outliers (GPU) - NEW PHASE
        print("阶段2.5: 异常点过滤 (GPU)")
        print("-" * 70)
        filter_start = time.time()
        original_size_before_filter = gpu_mappings.size()
        
        # Step 1: Remove extreme edge colors (often unreliable)
        # Scale margin based on bit depth (3 for 8-bit, ~768 for 16-bit)
        scaled_margin = int(round(3.0 * (self.max_val / 255.0)))
        gpu_mappings.remove_extreme_colors(margin=scaled_margin)
        
        # Step 2: Filter by Delta E (remove mappings with excessive color difference)
        # Delta E is in Oklab space, internally scaled from LAB threshold
        gpu_mappings.filter_by_delta_e(max_delta_e=50.0, percentile_threshold=97.0)
        
        # Step 3: Filter by local consistency (remove isolated outliers)
        # This removes noisy mappings that don't match their neighbors
        gpu_mappings.filter_outliers_by_local_consistency(grid_size=8, std_threshold=3.0)
        
        filter_time = time.time() - filter_start
        filtered_total = original_size_before_filter - gpu_mappings.size()
        filter_ratio = (filtered_total / original_size_before_filter) * 100 if original_size_before_filter > 0 else 0
        print(f"过滤完成: {original_size_before_filter:,} → {gpu_mappings.size():,}")
        print(f"总剔除: {filtered_total:,} 个异常点 ({filter_ratio:.1f}%, {filter_time:.2f}秒)\\n")
        
        # Compress (GPU)
        print("阶段3: 空间压缩 (GPU)")
        print("-" * 70)
        compress_start = time.time()
        original_size = gpu_mappings.size()
        
        # Scale threshold by bit depth (2.0 for 8-bit, ~512 for 16-bit)
        # This prevents 0% compression in 16-bit mode which would cause GPU OOM
        scaled_threshold = 3.0 * (self.max_val / 255.0)
        gpu_mappings.compress_spatial(threshold=scaled_threshold)
        
        compress_time = time.time() - compress_start
        compression_ratio = (1 - gpu_mappings.size() / original_size) * 100 if original_size > 0 else 0
        print(f"压缩完成: {original_size:,} → {gpu_mappings.size():,}")
        print(f"压缩率: {compression_ratio:.1f}% ({compress_time:.2f}秒)\\n")
        
        # Interpolate (GPU)
        print("阶段4: LUT插值 (GPU)")
        print("-" * 70)
        interp_start = time.time()
        
        # Generate grid (直接在GPU上)
        grid_points = self.generate_lut_grid()
        grid_tensor = torch.from_numpy(grid_points).to(self.device)
        
        # Interpolate using GPU tensors
        result_tensor = self.interpolate_gpu_tensor(grid_tensor, gpu_mappings)
        
        interp_time = time.time() - interp_start
        print(f"插值完成: {len(grid_tensor):,} 个网格点 ({interp_time:.1f}秒)\\n")
        
        # 阶段5: 单调性优化 (GPU)
        # print("阶段5: 单调性优化 (GPU)")
        # print("-" * 70)
        # mono_start = time.time()
        
        # # Reshape to 3D grid for spatial processing
        # lut_grid = result_tensor.reshape(self.lut_size, self.lut_size, self.lut_size, 3)
        # lut_grid = self.enforce_monotonicity_gpu(lut_grid)
        # result_tensor = lut_grid.reshape(-1, 3)
        
        # mono_time = time.time() - mono_start
        # print(f"单调性优化完成 ({mono_time:.2f}秒)\\n")
        
        # Only now download to CPU
        mapped_colors = result_tensor.cpu().numpy()
        
        # Clean up GPU
        gpu_mappings.clear_memory()
        del grid_tensor, result_tensor
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Convert to LUT format
        mapped_colors_norm = mapped_colors / float(self.max_val)
        mapped_colors_norm = np.clip(mapped_colors_norm, 0.0, 1.0)
        lut_data_3d = mapped_colors_norm.reshape(self.lut_size, self.lut_size, self.lut_size, 3)
        
        total_time = time.time() - start_time
        print(f"{'='*70}")
        print(f"✅ GPU-Native LUT生成完成!")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"性能提升: 零CPU-GPU中间传输")
        print(f"{'='*70}")
        
        self.lut_data = lut_data_3d
        return lut_data_3d
    
    def interpolate_gpu_tensor(self, grid_tensor: torch.Tensor, 
                                gpu_mappings: GPUColorMappings) -> torch.Tensor:
        """
        GPU-native interpolation using tensors (no CPU transfer)
        
        Args:
            grid_tensor: (N, 3) grid points on GPU
            gpu_mappings: GPUColorMappings with keys/values on GPU
            
        Returns:
            (N, 3) interpolated colors tensor on GPU
        """
        # Convert to Oklab (GPU)
        grid_ok = self.rgb_to_oklab_gpu(grid_tensor)
        keys_ok = self.rgb_to_oklab_gpu(gpu_mappings.keys_tensor)
        values_ok = self.rgb_to_oklab_gpu(gpu_mappings.values_tensor)
        
        # IDW interpolation (GPU)
        n_grid = len(grid_ok)
        n_mappings = len(keys_ok)
        result_ok = torch.zeros_like(grid_ok)
        
        # Use more neighbors for smoother result
        k = min(40, n_mappings)
        # Reduce batch_size to avoid GPU OOM, especially for 16-bit data processing
        batch_size = 1000
        
        print(f"  GPU插值: {n_grid:,} 点 × {n_mappings:,} 映射 (k={k})")
        
        for batch_idx in range((n_grid + batch_size - 1) // batch_size):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_grid)
            batch_points = grid_ok[start:end]
            
            # Calculate distances (GPU)
            distances = torch.cdist(batch_points, keys_ok, p=2)
            
            # Find k nearest
            topk_dists, topk_indices = torch.topk(distances, k=k, largest=False, dim=1)
            
            # IDW weights with p=2 (smoother than p=1)
            epsilon = 1e-6
            # Use squared distance for smoothness
            weights = 1.0 / (topk_dists.pow(2) + epsilon)
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            # Weighted sum
            batch_neighbors = values_ok[topk_indices]
            batch_result = (batch_neighbors * weights.unsqueeze(-1)).sum(dim=1)
            
            result_ok[start:end] = batch_result
            
            if (batch_idx + 1) % 5 == 0:
                progress = (batch_idx + 1) / ((n_grid + batch_size - 1) // batch_size)
                print(f"    进度: {progress:.1%}", end='\r')
        
        print()
        
        # Convert back to RGB (GPU)
        result_rgb = self.oklab_to_rgb_gpu(result_ok)
        
        return result_rgb

    def enforce_monotonicity_gpu(self, lut_grid: torch.Tensor) -> torch.Tensor:
        """
        Enforce monotonicity on the 3D LUT (GPU)
        
        Args:
            lut_grid: (S, S, S, 3) tensor on GPU
            
        Returns:
            (S, S, S, 3) tensor with enforced monotonicity
        """
        # LUT grid organization:
        # lut_grid[b, g, r, c]
        # Axis 0: Blue input (slowest)
        # Axis 1: Green input
        # Axis 2: Red input (fastest)
        # Channel 0: Red output
        # Channel 1: Green output
        # Channel 2: Blue output
        
        # We enforce that Output Channel X is monotonic with respect to Input Axis corresponding to X
        # R_out (ch 0) monotonic along R_in (axis 2)
        # G_out (ch 1) monotonic along G_in (axis 1)
        # B_out (ch 2) monotonic along B_in (axis 0)
        
        # Clone to avoid in-place modification issues during iteration
        result = lut_grid.clone()
        
        # Simple iterative smoothing that enforces monotonicity
        # Ensure R[i] >= R[i-1]
        
        # 1. Red Channel (0) along Red Axis (2)
        for r in range(1, result.shape[2]):
             result[:, :, r, 0] = torch.maximum(result[:, :, r, 0], result[:, :, r-1, 0])

        # 2. Green Channel (1) along Green Axis (1)
        for g in range(1, result.shape[1]):
            result[:, g, :, 1] = torch.maximum(result[:, g, :, 1], result[:, g-1, :, 1])

        # 3. Blue Channel (2) along Blue Axis (0)
        for b in range(1, result.shape[0]):
            result[b, :, :, 2] = torch.maximum(result[b, :, :, 2], result[b-1, :, :, 2])
            
        return result
