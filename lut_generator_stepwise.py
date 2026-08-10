import heapq
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Tuple, Dict, List, Optional

import numpy as np

from image_processor import ImageColorMapper


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

        
    def add_batch(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_weights: Optional[torch.Tensor] = None,
    ):
        """
        Add a batch of mappings (already on GPU)
        
        Args:
            new_keys: (M, 3) tensor of input RGB colors
            new_values: (M, 3) tensor of output RGB colors
            new_weights: (M,) tensor with sample counts/reliability weights
        """
        if new_weights is None:
            new_weights = torch.ones(len(new_keys), dtype=torch.float32, device=self.device)
        else:
            new_weights = new_weights.to(device=self.device, dtype=torch.float32)

        if self.keys_tensor is None:
            self.keys_tensor = new_keys
            self.values_tensor = new_values
            self.weights_tensor = new_weights
        else:
            # Concatenate tensors (GPU operation, no CPU transfer)
            self.keys_tensor = torch.cat([self.keys_tensor, new_keys], dim=0)
            self.values_tensor = torch.cat([self.values_tensor, new_values], dim=0)
            
            # Handle weights
            if self.weights_tensor is None:
                self.weights_tensor = torch.ones(len(self.keys_tensor) - len(new_keys), 
                                               dtype=torch.float32, device=self.device)
            self.weights_tensor = torch.cat([self.weights_tensor, new_weights], dim=0)

    
    def unique_and_merge(self, outlier_rejection=True):
        """
        Remove duplicate keys and average their values (all on GPU)
        
        Args:
            outlier_rejection: If True, perform a second pass to reject 
                              points that deviate too much from their group mean.
        """
        if self.keys_tensor is None:
            return
        
        # Encode RGB
        m1, m2, m3 = 1, self.multiplier, self.multiplier * self.multiplier
        encoded = (self.keys_tensor[:, 0].long() * m1 + 
                   self.keys_tensor[:, 1].long() * m2 + 
                   self.keys_tensor[:, 2].long() * m3)
        
        if self.weights_tensor is None:
            self.weights_tensor = torch.ones(len(self.keys_tensor), dtype=torch.float32, device=self.device)

        unique_encoded, inverse_indices = torch.unique(encoded, return_inverse=True)
        n_unique = len(unique_encoded)
        
        # First pass: calculate group means
        merged_weights = torch.zeros(n_unique, dtype=torch.float32, device=self.device)
        merged_weights.scatter_add_(0, inverse_indices, self.weights_tensor)
        
        merged_values = torch.zeros(n_unique, 3, dtype=torch.float32, device=self.device)
        weighted_values = self.values_tensor * self.weights_tensor.unsqueeze(1)
        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        merged_values.scatter_add_(0, indices_expanded, weighted_values)
        merged_values = merged_values / merged_weights.unsqueeze(1).clamp(min=1e-6)
        
        # Second pass: Outlier rejection within each unique color group
        if outlier_rejection:
            # Calculate distance of each point to its group mean
            group_means = merged_values[inverse_indices]
            # Use L1 distance for robust error estimation
            point_errors = torch.abs(self.values_tensor - group_means).max(dim=1).values
            
            # Simple threshold: if error > 10% of range, it's likely noise
            error_threshold = 0.1 * self.max_val
            valid_mask = point_errors < error_threshold
            
            if not valid_mask.all():
                # Re-calculate means with valid points only
                subset_keys = self.keys_tensor[valid_mask]
                subset_values = self.values_tensor[valid_mask]
                subset_weights = self.weights_tensor[valid_mask]
                subset_encoded = encoded[valid_mask]
                
                unique_encoded, inverse_indices = torch.unique(subset_encoded, return_inverse=True)
                n_unique = len(unique_encoded)
                
                merged_weights = torch.zeros(n_unique, dtype=torch.float32, device=self.device)
                merged_weights.scatter_add_(0, inverse_indices, subset_weights)
                
                merged_values = torch.zeros(n_unique, 3, dtype=torch.float32, device=self.device)
                indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
                merged_values.scatter_add_(0, indices_expanded, subset_values * subset_weights.unsqueeze(1))
                merged_values = merged_values / merged_weights.unsqueeze(1).clamp(min=1e-6)
        
        # Decode unique keys
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
    
    def downweight_local_inconsistency(self, grid_size: int = 6, strength: float = 0.7):
        """
        Reduce the influence of locally inconsistent mappings without deleting them.

        The score is based on output residuals in Oklab, not input/output distance.
        Coherent strong looks are preserved because a whole local region can move far
        from identity and still keep full weight.
        """
        if self.keys_tensor is None or len(self.keys_tensor) < 100:
            return

        if self.weights_tensor is None:
            self.weights_tensor = torch.ones(len(self.keys_tensor), dtype=torch.float32, device=self.device)

        grid_size_ok = grid_size / 100.0
        print(f"    局部残差一致性加权 (Oklab网格={grid_size_ok:.3f}, strength={strength})...")

        keys_ok = self.rgb_to_oklab_tensor(self.keys_tensor)
        values_ok = self.rgb_to_oklab_tensor(self.values_tensor)
        residuals_ok = values_ok - keys_ok

        keys_grid = torch.zeros_like(keys_ok, dtype=torch.long)
        keys_grid[:, 0] = (keys_ok[:, 0] / grid_size_ok).long().clamp(0, 255)
        keys_grid[:, 1] = ((keys_ok[:, 1] + 0.4) / grid_size_ok).long().clamp(0, 255)
        keys_grid[:, 2] = ((keys_ok[:, 2] + 0.4) / grid_size_ok).long().clamp(0, 255)
        cell_ids = keys_grid[:, 0] + keys_grid[:, 1] * 256 + keys_grid[:, 2] * 65536

        unique_cells, inverse_indices = torch.unique(cell_ids, return_inverse=True)
        n_cells = len(unique_cells)
        cell_count = torch.zeros(n_cells, dtype=torch.float32, device=self.device)
        cell_sum = torch.zeros(n_cells, 3, dtype=torch.float32, device=self.device)

        indices_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
        cell_count.scatter_add_(0, inverse_indices, torch.ones(len(residuals_ok), device=self.device))
        cell_sum.scatter_add_(0, indices_expanded, residuals_ok)
        cell_mean = cell_sum / cell_count.unsqueeze(1).clamp(min=1.0)

        point_mean = cell_mean[inverse_indices]
        residual_distance = torch.sqrt(torch.sum((residuals_ok - point_mean) ** 2, dim=1))

        cell_dist_sq_sum = torch.zeros(n_cells, dtype=torch.float32, device=self.device)
        cell_dist_sq_sum.scatter_add_(0, inverse_indices, residual_distance ** 2)
        cell_rms = torch.sqrt(cell_dist_sq_sum / cell_count.clamp(min=1.0))
        point_scale = cell_rms[inverse_indices].clamp(min=0.012)
        point_count = cell_count[inverse_indices]

        normalized = residual_distance / (point_scale * 3.0)
        robust_weight = 1.0 / (1.0 + normalized.pow(4))
        robust_weight = torch.where(
            point_count < 4,
            torch.ones_like(robust_weight),
            robust_weight.clamp(min=0.18),
        )

        old_weights = self.weights_tensor
        self.weights_tensor = old_weights * ((1.0 - strength) + strength * robust_weight)

        changed = torch.mean(torch.abs(self.weights_tensor - old_weights) / old_weights.clamp(min=1.0))
        print(
            f"    权重调整: 平均变化={float(changed) * 100:.2f}%, "
            f"最低保留权重={float(robust_weight.min()) * 100:.1f}%"
        )

        del keys_ok, values_ok, residuals_ok, keys_grid, cell_ids, cell_sum
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

    def output_is_grayscale(self) -> bool:
        """
        Detect monochrome targets from channel spread in output samples.

        A black-and-white target must stay on the neutral RGB axis throughout LUT
        generation. Otherwise residual interpolation and per-channel monotonic
        projection can reintroduce color.
        """
        if self.values_tensor is None or len(self.values_tensor) == 0:
            return False

        channel_spread = (
            self.values_tensor.max(dim=1).values - self.values_tensor.min(dim=1).values
        ).float()
        tolerance = max(3.0 * (self.max_val / 255.0), 1.0)
        p99 = float(torch.quantile(channel_spread, 0.99))
        mean = float(torch.mean(channel_spread))
        max_spread = float(torch.max(channel_spread))
        is_grayscale = p99 <= tolerance and mean <= tolerance
        status = "黑白/中性" if is_grayscale else "彩色"
        print(
            f"    目标色彩检测: {status} "
            f"(通道差 mean={mean:.3f}, P99={p99:.3f}, max={max_spread:.3f}, tol={tolerance:.3f})"
        )
        return is_grayscale

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

    INTERPOLATION_NEIGHBORS = 28
    INTERPOLATION_POWER = 2.4
    SMOOTH_ITERATIONS = 2
    SMOOTH_STRENGTH = 0.28

    def __init__(
        self,
        lut_size: int = 64,
        device: str = 'auto',
        bit_depth: int = 8,
    ):
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
        gpu_mappings.add_batch(unique_rgb_keys, mean_rgb, counts.float())

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
        print(f"\n{'='*70}")
        print(f"GPU-Native LUT生成 (零CPU传输模式)")
        print(f"{'='*70}")
        print(f"开始生成 {self.lut_size}x{self.lut_size}x{self.lut_size} 的3D LUT...")
        print(f"使用设备: {self.device.upper()} (GPU-Native流程)")
        
        start_time = time.time()
        
        # Find image pairs
        if os.path.isfile(photoa_dir) and os.path.isfile(photob_dir):
            image_pairs = [(photoa_dir, photob_dir)]
        else:
            photoa_files = sorted([os.path.join(photoa_dir, f) for f in os.listdir(photoa_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
            
            image_pairs = []
            for photoa_path in photoa_files:
                filename = os.path.basename(photoa_path)
                photob_path = os.path.join(photob_dir, filename)
                if os.path.exists(photob_path):
                    image_pairs.append((photoa_path, photob_path))
        
        if not image_pairs:
            raise ValueError("No matching image pairs found")

        print(f"找到 {len(image_pairs)} 对图片\n")
        
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
        if gpu_mappings.size() == 0:
            raise ValueError("No valid color mappings collected from image pairs")

        # Merge duplicates (GPU)
        print("阶段2: 合并重复 (GPU)")
        print("-" * 70)
        merge_start = time.time()
        gpu_mappings.unique_and_merge(outlier_rejection=False)
        merge_time = time.time() - merge_start
        print(f"合并完成: {gpu_mappings.size():,} 个唯一映射 ({merge_time:.2f}秒)\\n")

        target_is_grayscale = gpu_mappings.output_is_grayscale()
        print()
        
        print("阶段2.5: 局部一致性可靠度加权 (GPU)")
        print("-" * 70)
        weight_start = time.time()
        gpu_mappings.downweight_local_inconsistency(grid_size=6, strength=0.7)
        weight_time = time.time() - weight_start
        print(f"可靠度加权完成: {gpu_mappings.size():,} 个映射 ({weight_time:.2f}秒)\\n")

        print("阶段3: LUT尺度空间压缩 (GPU)")
        print("-" * 70)
        compress_start = time.time()
        original_size = gpu_mappings.size()

        # Match compression cell width to the final LUT grid spacing. This removes
        # redundant samples without averaging across visible LUT cells.
        grid_step = float(self.max_val) / max(self.lut_size - 1, 1)
        scaled_threshold = max(grid_step / 2.0, 0.5)
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
        result_tensor = self.interpolate_gpu_tensor(
            grid_tensor,
            gpu_mappings,
            force_monochrome=target_is_grayscale,
        )
        
        interp_time = time.time() - interp_start
        print(f"插值完成: {len(grid_tensor):,} 个网格点 ({interp_time:.1f}秒)\\n")
        
        print("阶段4.5: 保风格平滑 + 最小改动单调投影")
        print("-" * 70)
        constrain_start = time.time()
        lut_grid = result_tensor.reshape(self.lut_size, self.lut_size, self.lut_size, 3)
        identity_grid = grid_tensor.reshape(self.lut_size, self.lut_size, self.lut_size, 3)
        lut_grid = self.constrain_smooth_monotonic_gpu(
            lut_grid,
            identity_grid,
            force_monochrome=target_is_grayscale,
        )
        result_tensor = lut_grid.reshape(-1, 3)

        constrain_time = time.time() - constrain_start
        print(f"约束平滑完成 ({constrain_time:.2f}秒)\n")
        
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
    
    def interpolate_gpu_tensor(
        self,
        grid_tensor: torch.Tensor,
        gpu_mappings: GPUColorMappings,
        force_monochrome: bool = False,
    ) -> torch.Tensor:
        """
        GPU-native interpolation using tensors (no CPU transfer)
        
        Args:
            grid_tensor: (N, 3) grid points on GPU
            gpu_mappings: GPUColorMappings with keys/values on GPU
            force_monochrome: If True, constrain interpolation to neutral Oklab axis
            
        Returns:
            (N, 3) interpolated colors tensor on GPU
        """
        # Convert to Oklab (GPU)
        grid_ok = self.rgb_to_oklab_gpu(grid_tensor)
        keys_ok = self.rgb_to_oklab_gpu(gpu_mappings.keys_tensor)
        values_ok = self.rgb_to_oklab_gpu(gpu_mappings.values_tensor)
        if force_monochrome:
            values_ok = values_ok.clone()
            values_ok[:, 1:] = 0.0
        residuals_ok = values_ok - keys_ok
        
        # IDW interpolation (GPU)
        n_grid = len(grid_ok)
        n_mappings = len(keys_ok)
        result_ok = torch.zeros_like(grid_ok)
        
        k = min(self.INTERPOLATION_NEIGHBORS, n_mappings)
        power = self.INTERPOLATION_POWER
        mapping_weights = gpu_mappings.weights_tensor

        # Reduce batch_size to avoid GPU OOM, especially for 16-bit data processing
        batch_size = 1000
        
        weight_mode = "样本权重" if mapping_weights is not None else "均匀权重"
        color_mode = "黑白亮度残差" if force_monochrome else "彩色残差位移"
        print(f"  GPU插值: {n_grid:,} 点 × {n_mappings:,} 映射 (k={k}, p={power}, {color_mode}, {weight_mode})")
        
        for batch_idx in range((n_grid + batch_size - 1) // batch_size):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_grid)
            batch_points = grid_ok[start:end]
            
            # Calculate distances (GPU)
            distances = torch.cdist(batch_points, keys_ok, p=2)
            
            # Find k nearest
            topk_dists, topk_indices = torch.topk(distances, k=k, largest=False, dim=1)
            
            epsilon = 1e-6
            weights = 1.0 / (topk_dists.pow(power) + epsilon)
            if mapping_weights is not None:
                reliability = torch.sqrt(mapping_weights[topk_indices].clamp(min=1.0))
                weights = weights * reliability
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            # Weighted sum
            batch_neighbors = residuals_ok[topk_indices]
            batch_result = (batch_neighbors * weights.unsqueeze(-1)).sum(dim=1)
            batch_result = batch_points + batch_result
            if force_monochrome:
                batch_result[:, 1:] = 0.0
            
            result_ok[start:end] = batch_result
            
            if (batch_idx + 1) % 5 == 0:
                progress = (batch_idx + 1) / ((n_grid + batch_size - 1) // batch_size)
                print(f"    进度: {progress:.1%}", end='\r')
        
        print()
        
        # Convert back to RGB (GPU)
        result_rgb = self.oklab_to_rgb_gpu(result_ok)
        if force_monochrome:
            result_rgb = self.force_monochrome_rgb_gpu(result_rgb)
        
        return result_rgb

    def force_monochrome_rgb_gpu(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Put RGB values exactly on the neutral axis."""
        gray = (
            rgb_tensor[..., 0] * 0.2126
            + rgb_tensor[..., 1] * 0.7152
            + rgb_tensor[..., 2] * 0.0722
        )
        return gray.unsqueeze(-1).expand_as(rgb_tensor)

    def constrain_smooth_monotonic_gpu(
        self,
        lut_grid: torch.Tensor,
        identity_grid: torch.Tensor,
        force_monochrome: bool = False,
    ) -> torch.Tensor:
        """
        Preserve the learned look while enforcing smooth monotonic LUT behavior.

        1. Fit the raw LUT with L1 isotonic regression.
        2. Smooth only the learned residual, using edge-aware weights.
        3. Fit again so the final result is still monotonic.
        4. Blend with the initial L1 fit; convex combinations of
           monotonic LUTs remain monotonic and pull the result back toward style.
        """
        if force_monochrome:
            return self.constrain_monochrome_smooth_monotonic_gpu(lut_grid, identity_grid)

        print("    >> L1 Isotonic Regression 单调拟合 (不删点)...")
        l1_monotonic = self.enforce_monotonicity_gpu(lut_grid, verbose=True)
        current = l1_monotonic

        for iteration in range(self.SMOOTH_ITERATIONS):
            print(f"    >> 边缘保留残差平滑 {iteration + 1}/{self.SMOOTH_ITERATIONS}...")
            smoothed = self.edge_aware_smooth_residual_gpu(
                current,
                identity_grid,
                strength=self.SMOOTH_STRENGTH,
            )
            current = self.enforce_monotonicity_gpu(smoothed, verbose=False)

            # Pull a little toward the initial L1 fit to avoid over-smoothing
            # hue/style bends that were already valid after projection.
            current = current * 0.86 + l1_monotonic * 0.14

        return torch.clamp(current, 0, self.max_val)

    def constrain_monochrome_smooth_monotonic_gpu(
        self,
        lut_grid: torch.Tensor,
        identity_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Constrain a black-and-white LUT as one scalar gray field."""
        print("    >> 黑白LUT: 灰阶标量三轴 L1 Isotonic Regression...")
        gray_grid = self.rgb_luma_gpu(lut_grid)
        identity_gray = self.rgb_luma_gpu(identity_grid)
        l1_monotonic = self.enforce_scalar_monotonicity_gpu(gray_grid, verbose=True)
        current = l1_monotonic

        for iteration in range(self.SMOOTH_ITERATIONS):
            print(f"    >> 黑白LUT: 边缘保留灰阶残差平滑 {iteration + 1}/{self.SMOOTH_ITERATIONS}...")
            smoothed = self.edge_aware_smooth_scalar_residual_gpu(
                current,
                identity_gray,
                strength=self.SMOOTH_STRENGTH,
            )
            current = self.enforce_scalar_monotonicity_gpu(smoothed, verbose=False)
            current = current * 0.86 + l1_monotonic * 0.14

        current = torch.clamp(current, 0, self.max_val)
        return current.unsqueeze(-1).expand(*current.shape, 3)

    def rgb_luma_gpu(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Rec.709 luma used only for neutral-axis constraints."""
        return (
            rgb_tensor[..., 0] * 0.2126
            + rgb_tensor[..., 1] * 0.7152
            + rgb_tensor[..., 2] * 0.0722
        )

    def edge_aware_smooth_residual_gpu(
        self,
        lut_grid: torch.Tensor,
        identity_grid: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Smooth the LUT residual without blurring across strong style changes."""
        import torch.nn.functional as F

        residual = lut_grid - identity_grid
        x = residual.permute(3, 0, 1, 2).unsqueeze(0)
        center = x
        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')

        _, _, depth, height, width = x.shape
        accum = torch.zeros_like(x)
        weight_sum = torch.zeros((1, 1, depth, height, width), dtype=x.dtype, device=x.device)
        sigma_color = max(float(self.max_val) * 0.07, 1e-3)

        for db in (-1, 0, 1):
            for dg in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    neighbor = padded[
                        :,
                        :,
                        1 + db:1 + db + depth,
                        1 + dg:1 + dg + height,
                        1 + dr:1 + dr + width,
                    ]
                    spatial_dist_sq = float(db * db + dg * dg + dr * dr)
                    spatial_weight = float(np.exp(-spatial_dist_sq / 2.0))
                    residual_dist_sq = torch.sum((neighbor - center) ** 2, dim=1, keepdim=True)
                    range_weight = torch.exp(-residual_dist_sq / (2.0 * sigma_color * sigma_color))
                    weight = range_weight * spatial_weight
                    accum = accum + neighbor * weight
                    weight_sum = weight_sum + weight

        smoothed = accum / weight_sum.clamp(min=1e-6)
        x = center * (1.0 - strength) + smoothed * strength
        residual_smoothed = x.squeeze(0).permute(1, 2, 3, 0)
        return identity_grid + residual_smoothed

    def edge_aware_smooth_scalar_residual_gpu(
        self,
        gray_grid: torch.Tensor,
        identity_gray: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Edge-aware smoothing for a scalar monochrome LUT."""
        import torch.nn.functional as F

        residual = gray_grid - identity_gray
        x = residual.unsqueeze(0).unsqueeze(0)
        center = x
        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')

        _, _, depth, height, width = x.shape
        accum = torch.zeros_like(x)
        weight_sum = torch.zeros_like(x)
        sigma_gray = max(float(self.max_val) * 0.07, 1e-3)

        for db in (-1, 0, 1):
            for dg in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    neighbor = padded[
                        :,
                        :,
                        1 + db:1 + db + depth,
                        1 + dg:1 + dg + height,
                        1 + dr:1 + dr + width,
                    ]
                    spatial_dist_sq = float(db * db + dg * dg + dr * dr)
                    spatial_weight = float(np.exp(-spatial_dist_sq / 2.0))
                    residual_dist_sq = (neighbor - center) ** 2
                    range_weight = torch.exp(-residual_dist_sq / (2.0 * sigma_gray * sigma_gray))
                    weight = range_weight * spatial_weight
                    accum = accum + neighbor * weight
                    weight_sum = weight_sum + weight

        smoothed = accum / weight_sum.clamp(min=1e-6)
        x = center * (1.0 - strength) + smoothed * strength
        residual_smoothed = x.squeeze(0).squeeze(0)
        return identity_gray + residual_smoothed

    @staticmethod
    def _l1_isotonic_lower(seq: np.ndarray) -> np.ndarray:
        """Return the pointwise-lowest non-decreasing L1 isotonic fit."""
        values = np.asarray(seq, dtype=np.float64)
        result = np.empty(values.size, dtype=np.float64)
        max_heap = []

        # This is the heap/slope-trick form of unweighted L1 isotonic
        # regression. Duplicating each observation and removing the current
        # maximum maintains the lower median of every active prefix.
        for idx, value in enumerate(values):
            value = float(value)
            heapq.heappush(max_heap, -value)
            heapq.heappush(max_heap, -value)
            heapq.heappop(max_heap)
            result[idx] = -max_heap[0]

        # Prefix medians may decrease. The reverse cumulative minimum turns
        # them into the pointwise-lowest optimal isotonic sequence.
        for idx in range(result.size - 2, -1, -1):
            result[idx] = min(result[idx], result[idx + 1])

        return result

    @classmethod
    def _l1_isotonic_non_decreasing(cls, seq: np.ndarray) -> np.ndarray:
        """Minimize sum(abs(fit - seq)) subject to a non-decreasing fit.

        L1 isotonic regression can have multiple optima. The returned fit is
        the midpoint of the pointwise-lowest and pointwise-highest solutions,
        avoiding a systematic dark/bright bias while retaining minimum L1
        error.
        """
        values = np.asarray(seq, dtype=np.float64)
        if values.size == 0:
            return np.empty_like(values, dtype=np.float32)

        lower = cls._l1_isotonic_lower(values)
        upper = -cls._l1_isotonic_lower(-values[::-1])[::-1]
        return ((lower + upper) * 0.5).astype(np.float32)

    @staticmethod
    def _max_monotonic_violation_3d(grid_np: np.ndarray) -> float:
        """Largest downward step along any input axis."""
        violations = [
            -np.min(np.diff(grid_np, axis=0)),
            -np.min(np.diff(grid_np, axis=1)),
            -np.min(np.diff(grid_np, axis=2)),
        ]
        return float(max(0.0, *violations))

    def enforce_scalar_monotonicity_gpu(
        self,
        gray_grid: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Fit a scalar LUT monotonically on all input axes with L1 loss."""
        device = gray_grid.device
        grid_np = gray_grid.detach().cpu().numpy().astype(np.float32, copy=True)
        original_np = grid_np.copy()

        for _ in range(8):
            for b in range(grid_np.shape[0]):
                for g in range(grid_np.shape[1]):
                    grid_np[b, g, :] = self._l1_isotonic_non_decreasing(grid_np[b, g, :])

            for b in range(grid_np.shape[0]):
                for r in range(grid_np.shape[2]):
                    grid_np[b, :, r] = self._l1_isotonic_non_decreasing(grid_np[b, :, r])

            for g in range(grid_np.shape[1]):
                for r in range(grid_np.shape[2]):
                    grid_np[:, g, r] = self._l1_isotonic_non_decreasing(grid_np[:, g, r])

            if self._max_monotonic_violation_3d(grid_np) <= 1e-4:
                break

        remaining_violation = self._max_monotonic_violation_3d(grid_np)
        if remaining_violation > 1e-4:
            grid_np = np.maximum.accumulate(grid_np, axis=2)
            grid_np = np.maximum.accumulate(grid_np, axis=1)
            grid_np = np.maximum.accumulate(grid_np, axis=0)

        if verbose:
            correction = np.abs(grid_np - original_np)
            print(
                f"    灰阶 L1 单调拟合修正: 平均={correction.mean():.3f}, "
                f"P95={np.percentile(correction, 95):.3f}, 最大={correction.max():.3f}"
            )

        return torch.from_numpy(grid_np).to(device)

    def enforce_monotonicity_gpu(self, lut_grid: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """Fit each primary channel axis monotonically with minimum L1 error."""
        device = lut_grid.device
        grid_np = lut_grid.detach().cpu().numpy()
        original_np = grid_np.copy()

        for b in range(grid_np.shape[0]):
            for g in range(grid_np.shape[1]):
                grid_np[b, g, :, 0] = self._l1_isotonic_non_decreasing(grid_np[b, g, :, 0])

        for b in range(grid_np.shape[0]):
            for r in range(grid_np.shape[2]):
                grid_np[b, :, r, 1] = self._l1_isotonic_non_decreasing(grid_np[b, :, r, 1])

        for g in range(grid_np.shape[1]):
            for r in range(grid_np.shape[2]):
                grid_np[:, g, r, 2] = self._l1_isotonic_non_decreasing(grid_np[:, g, r, 2])

        if verbose:
            correction = np.abs(grid_np - original_np)
            print(
                f"    L1 单调拟合修正: 平均={correction.mean():.3f}, "
                f"P95={np.percentile(correction, 95):.3f}, 最大={correction.max():.3f}"
            )

        return torch.from_numpy(grid_np).to(device)
