from transformers import AutoImageProcessor, EfficientNetModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch


class EfficientNet(VectorizationFunction):
    def __init__(self):
        """
        Initializes the EfficientNet-based vectorization function with the EfficientNet model and processor.
        """
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        super().__init__(self.vectorize_image)

    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        """
        Processes the image and extracts features using the EfficientNet model.

        Args:
            image (PIL.Image): The input preprocessed image.

        Returns:
            np.ndarray: The feature vector extracted by the EfficcientNet model.
        """
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = outputs.last_hidden_state.squeeze(0).numpy()
        
        # uncomment to have a normalized embedding of size (2560,) instead of (2560, 19, 19)
        embedding = np.mean(embedding, axis=(1, 2))
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
