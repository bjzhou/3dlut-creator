from PIL import Image
import os
import glob

files = glob.glob("fujia/*")
if files:
    img = Image.open(files[0])
    print(f"File: {files[0]}")
    print(f"Mode: {img.mode}")
    print(f"Format: {img.format}")
    import numpy as np
    data = np.array(img)
    print(f"Dtype: {data.dtype}")
    print(f"Max: {data.max()}")
else:
    print("No files found in fujia")
