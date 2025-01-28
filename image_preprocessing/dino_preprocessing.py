from image_feature_extraction import ImagePreprocessingFunction
import numpy as np
from PIL import Image

class DinoPreprocessor(ImagePreprocessingFunction):
    def __init__(self):
        super().__init__(self.preprocess_image)

    def preprocess_image(self, img : Image):
        return img