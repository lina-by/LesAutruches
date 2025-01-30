from transformers import CLIPProcessor, CLIPModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class CLIP(VectorizationFunction):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP-based vectorization function with the CLIP model and processor.
        Possible model names are:
        -   openai/clip-vit-base-patch32
        -   openai/clip-vit-base-patch16
        -   openai/clip-vit-large-patch14
        """
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        super().__init__(self.vectorize_image)

    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        """
        Processes the image and extracts features using the CLIP model.

        Args:
            image (PIL.Image): The input preprocessed image.

        Returns:
            np.ndarray: The feature vector extracted by the CLIP model.
        """
        inputs = self.processor(
            text = None,
            images = image,
            return_tensors="pt"
            )["pixel_values"]
        
        outputs = self.model.get_image_features(inputs)
        
        embedding =  outputs.cpu().detach().numpy()
        
        # uncomment to get normalized embedding with shape (768,) instead of (1, 768)
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
