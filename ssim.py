import os
import re
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def get_image_number(file_name):
    return int(re.findall(r'\d+', file_name)[-1])

def calculate_ssim(img1, img2):
    # Convert images to grayscale
    img1 = np.array(img1.convert('L'))
    img2 = np.array(img2.convert('L'))

    # Calculate SSIM
    return ssim(img1, img2)

def main():
    dir1 = 'experiments/gap_2/interpolated/'
    dir2 = 'experiments/gap_8/interpolated/'
    ssim_values = []

    for file1 in sorted(os.listdir(dir1)):
        num1 = get_image_number(file1)
        print(f"Processing {file1} with num {num1}")

        file2 = [f for f in os.listdir(dir2) if get_image_number(f) == num1]
        if file2:
            file2 = file2[0]
            print(f"Matching file found: {file2}")

            img1 = Image.open(os.path.join(dir1, file1))
            img2 = Image.open(os.path.join(dir2, file2))

            ssim_val = calculate_ssim(img1, img2)
            ssim_values.append(ssim_val)
        else:
            print(f"No matching file for {file1}")

    if ssim_values:
        average_ssim = sum(ssim_values) / len(ssim_values)
        print(f"Average SSIM: {average_ssim}")
    else:
        print("No SSIM values calculated.")

if __name__ == "__main__":
    main()
