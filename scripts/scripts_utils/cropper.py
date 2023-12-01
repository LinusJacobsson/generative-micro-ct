import os
from PIL import Image

def crop_images(input_dir, output_dir, initial_offset_x=0, initial_offset_y=0, change_after=340, new_offset_x=-100, new_offset_y=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    offset_x, offset_y = initial_offset_x, initial_offset_y
    count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            if count >= change_after:
                offset_x, offset_y = new_offset_x, new_offset_y

            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            center_x, center_y = img.width // 2, img.height // 2
            center_x += offset_x
            center_y += offset_y

            left = max(center_x - 256, 0)
            top = max(center_y - 256, 0)
            right = min(center_x + 256, img.width)
            bottom = min(center_y + 256, img.height)

            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(os.path.join(output_dir, filename))

            count += 1

# Example usage
input_folder_path = '../../cropped_medium_res/'
output_folder_path = '../../adaptive_boxes/'

# Initial offset values
initial_offset_x = 0
initial_offset_y = 0

# New offset values after 'change_after' number of images
new_offset_x = -100  # Move more to the left
new_offset_y = 0
change_after = 340  # Change offset after this many images

crop_images(input_folder_path, output_folder_path, initial_offset_x, initial_offset_y, change_after, new_offset_x, new_offset_y)
