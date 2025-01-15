from transformers import pipeline
from image_feature_extraction import ImagePreprocessingFunction
import numpy as np
from PIL import Image

class ImageSegmentationPreprocessor(ImagePreprocessingFunction):
    def __init__(self):
        """
        Initializes the segmentation-based preprocessing function with the briaai segmentation pipeline.
        """
        self.segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
        super().__init__(self.preprocess_image)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Applies segmentation to the input image to extract the object of interest.

        Args:
            image (PIL.Image): The input image.

        Returns:
            PIL.Image: The segmented image with the background removed or object of interest highlighted.
        """
        binary_mask = np.array(self.segmentation_pipeline(image, return_mask=True)) * 255
        binary_mask = Image.fromarray(binary_mask).convert("L")

        # Apply the mask to the original image
        image_array = np.array(image)  # Convert original image to a NumPy array
        black_background = np.zeros_like(image_array)  # Create a black background
        segmented_image_array = np.where(np.expand_dims(binary_mask, axis=-1) > 0, image_array, black_background)

        # Convert back to a PIL Image
        segmented_image = Image.fromarray(segmented_image_array)
        return segmented_image