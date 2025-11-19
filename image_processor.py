import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import glob


class ImageColorMapper:
    """RGB color mapper for images"""

    def __init__(self, photoa_dir: str, photob_dir: str):
        """
        Initialize color mapper

        Args:
            photoa_dir: Base RGB images directory (input colors)
            photob_dir: Mapped RGB images directory (output colors)
        """
        self.photoa_dir = photoa_dir
        self.photob_dir = photob_dir
        self.color_mappings: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}

    def load_image_pairs(self) -> List[Tuple[str, str]]:
        """
        Get paired image file paths

        Returns:
            List of paired image paths [(photoa_path, photob_path), ...]
        """
        # Get all image files in photoa directory
        photoa_files = glob.glob(os.path.join(self.photoa_dir, "*"))
        photoa_files = [f for f in photoa_files if os.path.isfile(f) and
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        pairs = []
        for photoa_path in photoa_files:
            filename = os.path.basename(photoa_path)
            photob_path = os.path.join(self.photob_dir, filename)

            if os.path.exists(photob_path):
                pairs.append((photoa_path, photob_path))
            else:
                print(f"Warning: Corresponding file not found in photob directory {filename}")

        return pairs

    def extract_rgb_data(self, image_path: str) -> np.ndarray:
        """
        Extract RGB data from image

        Args:
            image_path: Image file path

        Returns:
            RGB data array with shape (height, width, 3)
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            rgb_data = np.array(image)
            return rgb_data

        except Exception as e:
            print(f"Failed to read image {image_path}: {e}")
            return np.array([])

    def collect_color_mappings(self):
        """
        Collect color mapping relationships from all image pairs
        """
        image_pairs = self.load_image_pairs()

        if not image_pairs:
            raise ValueError("No valid image pairs found")

        print(f"Found {len(image_pairs)} image pairs")

        for i, (photoa_path, photob_path) in enumerate(image_pairs):
            print(f"Processing pair {i+1}: {os.path.basename(photoa_path)}")

            # Extract RGB data
            rgb_a = self.extract_rgb_data(photoa_path)
            rgb_b = self.extract_rgb_data(photob_path)

            if rgb_a.size == 0 or rgb_b.size == 0:
                continue

            # Ensure image dimensions match
            if rgb_a.shape != rgb_b.shape:
                print(f"Warning: Image dimensions mismatch, skipping {os.path.basename(photoa_path)}")
                continue

            # Collect color mapping relationships
            height, width = rgb_a.shape[:2]
            for y in range(height):
                for x in range(width):
                    rgb_in = tuple(rgb_a[y, x])  # Base RGB
                    rgb_out = tuple(rgb_b[y, x])  # Mapped RGB

                    if rgb_in not in self.color_mappings:
                        self.color_mappings[rgb_in] = []

                    self.color_mappings[rgb_in].append(rgb_out)

        print(f"Collected {len(self.color_mappings)} different input colors")

        # Average multiple mapping values for each input color
        self._average_mappings()

    def _average_mappings(self):
        """
        Average multiple mapping values for each input color
        """
        print("Calculating color mapping averages...")
        averaged_mappings = {}

        for rgb_in, rgb_out_list in self.color_mappings.items():
            if rgb_out_list:
                avg_rgb = tuple(int(round(np.mean(rgb_out_list, axis=0)[i])) for i in range(3))
                averaged_mappings[rgb_in] = avg_rgb

        self.color_mappings = averaged_mappings
        print(f"Average calculation completed, {len(self.color_mappings)} mapping relationships remaining")

    def get_color_mapping(self, rgb: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """
        Get mapping for specific RGB color

        Args:
            rgb: Input RGB value

        Returns:
            Mapped RGB value, or None if not found
        """
        return self.color_mappings.get(rgb)


def test_image_processor():
    """Test image processor functionality"""
    # Create test directory structure
    os.makedirs("photoa", exist_ok=True)
    os.makedirs("photob", exist_ok=True)

    # Create test images (simple gradients)
    from PIL import Image, ImageDraw

    # Generate test images
    for i in range(3):  # Create 3 pairs of test images
        # photoa: red gradient
        img_a = Image.new('RGB', (100, 100))
        draw_a = ImageDraw.Draw(img_a)
        for x in range(100):
            color = (x * 255 // 100, i * 50, 100)
            draw_a.line([(x, 0), (x, 100)], fill=color)
        img_a.save(f"photoa/test_{i}.png")

        # photob: blue gradient (simulating color mapping)
        img_b = Image.new('RGB', (100, 100))
        draw_b = ImageDraw.Draw(img_b)
        for x in range(100):
            color = (50, 50, x * 255 // 100)
            draw_b.line([(x, 0), (x, 100)], fill=color)
        img_b.save(f"photob/test_{i}.png")

    # Test processor
    processor = ImageColorMapper("photoa", "photob")
    processor.collect_color_mappings()

    print(f"Successfully collected {len(processor.color_mappings)} color mappings")

    # Show some mapping results
    for i, (rgb_in, rgb_out) in enumerate(list(processor.color_mappings.items())[:5]):
        print(f"Mapping {i+1}: {rgb_in} -> {rgb_out}")


if __name__ == "__main__":
    test_image_processor()