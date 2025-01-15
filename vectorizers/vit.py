from transformers import ViTImageProcessor, ViTModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class Vit(VectorizationFunction):
    def __init__(self):
        """
        Initializes the ViT-based vectorization function with the ViT model and processor.
        """
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        super().__init__(self.vectorize_image)

    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        """
        Processes the image and extracts features using the ViT model.

        Args:
            image (PIL.Image): The input preprocessed image.

        Returns:
            np.ndarray: The feature vector extracted by the ViT model.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0).numpy()
