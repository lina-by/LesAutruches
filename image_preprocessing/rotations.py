from PIL import Image, ExifTags
import numpy as np
from image_feature_extraction import ImagePreprocessingFunction


def fix_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            orientation_tag = next((tag for tag, value in ExifTags.TAGS.items() if value == 'Orientation'), None)

            if orientation_tag and orientation_tag in exif:
                orientation = exif[orientation_tag]

                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                else:
                    return image
                return Image.fromarray(np.array(image))

    except Exception as e:
        print(f"Erreur lors de la correction d'orientation : {e}")
    
    return image


class OrientationImages(ImagePreprocessingFunction):
    def __init__(self):
        """
        Initializes the segmentation-based preprocessing function with the briaai segmentation pipeline.
        """
        super().__init__(fix_image_orientation)