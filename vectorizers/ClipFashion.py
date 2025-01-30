from transformers import CLIPProcessor, CLIPModel
from image_feature_extraction import VectorizationFunction
from PIL import Image
import numpy as np
import torch

class FashionClip(VectorizationFunction):
    def __init__(self):
        """
        Initialise la classe FashionClip avec le modèle et le processeur Fashion-CLIP.
        """
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        super().__init__(self.vectorize_image)
    
    def vectorize_image(self, image: Image.Image) -> np.ndarray:
        """
        Extrait les features d'une image en utilisant Fashion-CLIP.

        Args:
            image (PIL.Image): L'image d'entrée à vectoriser.

        Returns:
            np.ndarray: Le vecteur d'embedding de l'image.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalisation de l'embedding
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_embedding.squeeze().numpy()
