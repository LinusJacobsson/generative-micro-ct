import imageio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb


def segment_and_crop(image_path, downsample_factor, expansion_margin):

    # Läs in bilden
    image = imageio.imread(image_path)

    # Minska bildens storlek
    downsampled_image = image[::downsample_factor, ::downsample_factor]

    # Använd Otsu thresholding
    thresh = threshold_otsu(downsampled_image)
    bw = downsampled_image > thresh

    # Ta bort artefakter kopplade till bildkanten och stäng luckor
    cleared = clear_border(bw)
    closed = closing(cleared, square(3))

    # Hitta den största regionen, alltså tumören
    label_image = label(closed)
    regions = regionprops(label_image)
    tumor_region = max(regions, key=lambda x: x.area)

    # Beräkna områdesramen för tumörregionen
    tumor_bounding_box = tumor_region.bbox

    # Skala upp områdesramen till den ursprungliga bildstorleken
    scaled_bounding_box = tuple(coord * downsample_factor for coord in tumor_bounding_box)

    # Expandera områdesramen (för att inte råka skära bort delar)
    expanded_bounding_box = (
        max(scaled_bounding_box[0] - expansion_margin, 0),
        max(scaled_bounding_box[1] - expansion_margin, 0),
        min(scaled_bounding_box[2] + expansion_margin, image.shape[0]),
        min(scaled_bounding_box[3] + expansion_margin, image.shape[1])
    )

    # Beskär tumörregionen från den ursprungliga högupplösta bilden
    tumor_cropped = image[expanded_bounding_box[0]:expanded_bounding_box[2], 
                          expanded_bounding_box[1]:expanded_bounding_box[3]]

    return tumor_cropped


image_path = '/Users/linus/Desktop/cropped_medium_res/bin2_174_tumor_Nr56_x4_StitchPag_bin2_stitch_1282x2191x10810004.tif'
downsample_factor = 10
expansion_margin = 5 * downsample_factor # testa runt. För att fånga in tumören precis

# Segmentera och beskär bilden
cropped_image = segment_and_crop(image_path, downsample_factor, expansion_margin)


#cropped_image_path = ''
#imageio.imwrite(cropped_image_path, cropped_image)

plt.imshow(cropped_image, cmap='gray')
plt.axis('off')
plt.show()
