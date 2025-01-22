# https://huggingface.co/docs/transformers/model_doc/resnet

from transformers import AutoImageProcessor, ResNetModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class ResNet(VectorizationFunction):
    def __init__(self):
        """
        Initializes the ResNet-based vectorization function with the ResNet model and processor.
        """
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        super().__init__(self.vectorize_image)

    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        """
        Processes the image and extracts features using the ResNet model.

        Args:
            image (PIL.Image): The input preprocessed image.

        Returns:
            np.ndarray: The feature vector extracted by the RestNet model.
        """
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = outputs.last_hidden_state.squeeze(0).numpy()
        
        # uncomment to have a normalized embedding of size (2048,) instead of (2048, 7, 7)
        # embedding = np.mean(embedding, axis=(1, 2))
        # embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    

