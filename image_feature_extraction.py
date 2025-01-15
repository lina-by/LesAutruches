from PIL import Image
import os
import numpy as np
from typing import Callable

class ImagePreprocessingFunction:
    """
    Represents an image preprocessing function.
    Takes a PIL Image as input, processes it, and returns a PIL Image.
    Optionally, it may return hyperparameters or additional information.
    """
    def __init__(self, preprocess_function: Callable[[Image.Image], Image.Image]):
        self.preprocess_function = preprocess_function

    def __call__(self, img: Image.Image):
        return self.preprocess_function(img)

class VectorizationFunction:
    """
    Represents a vectorization function.
    Takes a PIL Image as input and returns a feature vector (e.g., numpy array).
    """
    def __init__(self, vectorize_function: Callable[[Image.Image], np.ndarray]):
        self.vectorize_function = vectorize_function

    def __call__(self, img: Image.Image):
        return self.vectorize_function(img)

class ExtractFeatureMethod:
    """
    Pipeline for extracting features from images.
    Combines preprocessing and vectorization to extract features from:
    - A single image
    - A list of images
    - A folder or list of image paths
    """
    def __init__(self, preprocessing_function: ImagePreprocessingFunction, vectorization_function: VectorizationFunction):
        self.preprocessing_function = preprocessing_function
        self.vectorization_function = vectorization_function

    def run_on_image(self, img: Image.Image):
        """
        Runs the pipeline on a single PIL Image.
        """
        preprocessed_img = self.preprocessing_function(img)
        feature_vector = self.vectorization_function(preprocessed_img)
        return feature_vector

    def run_on_images(self, images: list[Image.Image]):
        """
        Runs the pipeline on a list of PIL Images.
        """
        feature_vectors = [self.run_on_image(img) for img in images]
        return feature_vectors

    def run_on_paths(self, paths: list[str]):
        """
        Runs the pipeline on a list of image paths or a directory containing images.
        """
        images = []
        for path in paths:
            if os.path.isdir(path):
                # Process all images in the folder
                for file_name in os.listdir(path):
                    file_path = os.path.join(path, file_name)
                    if self._is_image_file(file_path):
                        images.append(Image.open(file_path))
            elif os.path.isfile(path) and self._is_image_file(path):
                images.append(Image.open(path))
        
        return self.run_on_images(images)
    

    @staticmethod
    def _is_image_file(file_path: str):
        """
        Checks if a file path corresponds to an image.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return os.path.splitext(file_path)[1].lower() in valid_extensions
