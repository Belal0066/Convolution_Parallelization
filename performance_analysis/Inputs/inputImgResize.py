import numpy as np
from PIL import Image

# Constants
INPUT_FILE = "waterfall_grey_1920_2520.raw"
INPUT_SIZE = (1920, 2520)  # (width, height)
OUTPUT_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]

# Load raw grayscale image
with open(INPUT_FILE, "rb") as f:
    raw_data = f.read()

# Convert to NumPy array and reshape (note: height, width)
image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((INPUT_SIZE[1], INPUT_SIZE[0]))

# Convert to PIL Image (mode 'L' for grayscale)
image = Image.fromarray(image_array, mode='L')

# Resize and save as raw
for size in OUTPUT_SIZES:
    resized = image.resize(size, Image.LANCZOS)
    resized_array = np.array(resized, dtype=np.uint8)
    output_filename = f"input_blur_waterfall_grey_{size[0]}x{size[1]}.raw"
    with open(output_filename, "wb") as f:
        f.write(resized_array.tobytes())
    print(f"Saved: {output_filename}")
