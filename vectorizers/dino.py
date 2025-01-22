from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class Dino(VectorizationFunction):
    def __init__(self, model_size: str = "base"):
        """
        Initializes the ViT-based vectorization function with the dinov2 model and processor.
        
        Args:
            model_size: "small" or "base" to choose the size of the model
        """
        self.processor = AutoImageProcessor.from_pretrained(f'facebook/dinov2-{model_size}', use_fast=True)
        self.model = AutoModel.from_pretrained(f'facebook/dinov2-{model_size}')
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

        last_hidden_state = outputs.last_hidden_state.squeeze(0)
        cls_embedding = last_hidden_state[0, :]
        avg_embedding = last_hidden_state[1:, :].mean(dim = 0)
        return torch.cat((cls_embedding, avg_embedding), dim=0)
