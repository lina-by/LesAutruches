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
        """
        Applies segmentation to the input image to extract the object of interest.

        Args:
            image (PIL.Image): The input image.

        Returns:
            PIL.Image: The segmented image with the background removed or object of interest highlighted.
        """
        if self.rotations:
            image=fix_image_orientation(image)
        pillow_image = self.segmentation_pipeline(image)
        array = np.array(pillow_image)
        mask = array[:, :, 3]==0

        # Keep only RGB channels & Apply mask
        array = array[:, :, :3]
        array[mask, :] = [255, 255, 255]

        #TODO: zoomer sur l'objet en se servant du mask
        return Image.fromarray(array)