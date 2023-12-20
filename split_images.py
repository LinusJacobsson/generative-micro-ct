from PIL import Image, ImageDraw
import os

def add_borders_and_save(image, save_path, border_width=2):
    channels = image.split()

    total_width = image.width * 3 + border_width * 2
    new_img = Image.new('L', (total_width, image.height))

    for i, channel in enumerate(channels):
        position = i * image.width + border_width * i
        new_img.paste(channel, (position, 0))

    draw = ImageDraw.Draw(new_img)
    for i in range(1, 3):
        line_pos = i * image.width + border_width * (i - 1)
        draw.line([(line_pos, 0), (line_pos, image.height)], fill=0, width=border_width)

    new_img.save(save_path)

def process_directory(input_directory, output_directory, border_width=2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            file_path = os.path.join(input_directory, filename)
            img = Image.open(file_path)
            save_path = os.path.join(output_directory, 'processed_' + filename)
            add_borders_and_save(img, save_path, border_width)


# Usage
#input_directory = '/cephyr/users/linusjac/Alvis/image-restoration-sde/codes/config/deraining/result/ir-sde/Test_Dataset'  # Replace with your directory path
input_directory = 'experiments/gap_3'
output_directory = 'gap_3/'  # Replace with your output directory path
process_directory(input_directory, output_directory)
