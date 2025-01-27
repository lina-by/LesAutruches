from transformers import pipeline
from image_feature_extraction import ImagePreprocessingFunction
import numpy as np
from PIL import Image
from image_preprocessing.rotations import fix_image_orientation

class ImageSegmentationPreprocessor(ImagePreprocessingFunction):
    def __init__(self, rotations=False):
        """
        Initializes the segmentation-based preprocessing function with the briaai segmentation pipeline.
        """
        self.segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
        self.rotations=rotations
        super().__init__(self.preprocess_image)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if self.rotations:
            image = fix_image_orientation(image)
        pillow_image = self.segmentation_pipeline(image)
        array = np.array(pillow_image)
        mask = array[:, :, 3] == 0

        array = array[:, :, :3]
        array[mask, :] = [255, 255, 255]  # Set background pixels to white

        # Find the bounding box of the object by using the mask
        non_empty_pixels = np.where(~mask)
        top_percentile = np.percentile(non_empty_pixels[0], 1)
        bottom_percentile = np.percentile(non_empty_pixels[0], 99)
        left_percentile = np.percentile(non_empty_pixels[1], 1)
        right_percentile = np.percentile(non_empty_pixels[1], 99)

        # Convert to integers and create bounding box
        top_bottom = int(0.05*(bottom_percentile-top_percentile))
        left_right = int(0.05*(right_percentile-left_percentile))
        top = max(0, int(top_percentile) - top_bottom)  # Ensure top is not less than 0
        bottom = min(array.shape[0] - 1, int(bottom_percentile) + top_bottom)  # Ensure bottom is within image height
        left = max(0, int(left_percentile) - left_right)  # Ensure left is not less than 0
        right = min(array.shape[1] - 1, int(right_percentile) + left_right)  # Ensure right is within image width

        # Crop the image using the bounding box
        cropped_image = array[top:bottom + 1, left:right + 1] 

        # Convert cropped array back to PIL Image
        cropped_image_pil = Image.fromarray(cropped_image)

        return cropped_image_pil
