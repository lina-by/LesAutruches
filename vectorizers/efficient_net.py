from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class EfficientNet(VectorizationFunction):
    def __init__(self):
        """
        Initializes the ViT-based vectorization function with the ViT model and processor.
        """
        self.model = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
        super().__init__(self.vectorize_image)

    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        
        inputs = np.expand_dims(image, axis=0)
        with torch.no_grad():
            embedding = self.model.predict(inputs)
        return embedding.flatten()