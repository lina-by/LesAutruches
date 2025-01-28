from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    
    
    def _process_and_save(self, img_path: str, save_dir: str, save_format: str):
        """Applies preprocessing and saves the image in the given directory."""
        image = Image.open(img_path)

        # Ensure image is in a compatible mode
        if image.mode in ("RGBA", "P"):  # Convert RGBA/P images to RGB if necessary
            image = image.convert("RGB")

        processed_image = self.preprocess_function(image)

        # Generate save path with correct format
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f"{image_name}.{save_format}")

        # Save the processed image
        processed_image.save(save_path, format=save_format.upper())


    def run_on_paths(self, paths: list[str], save_folder_name: str, save_format="png"):
        """
        Runs the pipeline on a list of image paths or directories containing images.
        Saves the processed images in 'data/preprocessed/save_folder_name'.
        Creates subfolders named after the folder names in the paths.

        Args:
            paths (list[str]): List of image file paths or directories containing images.
            save_folder_name (str): Folder name to save the processed images.
            save_format (str): Format to save images (default: "png").
        """
        # Create the save folder if it doesn't exist
        save_folder = os.path.join("data/preprocessed", save_folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for path in paths:
            if os.path.isdir(path):
                folder_name = os.path.basename(os.path.normpath(path))  # Get the folder name
                folder_save_path = os.path.join(save_folder, folder_name)

                # Create subfolder if it doesn't exist
                os.makedirs(folder_save_path, exist_ok=True)

                # Process all images in the folder
                for file_name in os.listdir(path):
                    file_path = os.path.join(path, file_name)
                    if ExtractFeatureMethod._is_image_file(file_path):
                        self._process_and_save(file_path, folder_save_path, save_format)
            elif os.path.isfile(path) and ExtractFeatureMethod._is_image_file(path):
                self._process_and_save(path, save_folder, save_format)

        print(f"Images saved in folder: {save_folder}")

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

    def __call__(self, img: Image.Image):
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
        feature_vectors = [self.run_on_image(img) for img in tqdm(images)]
        return feature_vectors


    def run_on_paths(self, paths: list[str], save_folder_name: str, save_unique_df: bool = False):
        """
        Runs the pipeline on a list of image paths or a directory containing images.
        Saves the embeddings in embeddings/save_folder_name.
        Creates subfolders named after the folder names in the paths.

        Args:
            paths (list[str]): List of image file paths or directories containing images.
            save_folder_name (str): Folder name to save the embeddings.
            save_unique_df (bool): if True, the output is saved in a unique pandas dataframe
        """
        # Create the save folder if it doesn't exist
        save_folder = os.path.join("embeddings", save_folder_name)
        os.makedirs(save_folder, exist_ok=True)

        images = []
        image_paths = []
        for path in paths:
            if os.path.isdir(path):
                print(path)
                folder_name = os.path.basename(os.path.normpath(path))  # Get the folder name
                folder_save_path = os.path.join(save_folder, folder_name)

                # Create subfolder if it doesn't exist
                os.makedirs(folder_save_path, exist_ok=True)

                # Process all images in the folder
                for file_name in tqdm(os.listdir(path)):
                    file_path = os.path.join(path, file_name)
                    if self._is_image_file(file_path):
                        images.append(Image.open(file_path))
                        # Save images in subfolder with folder structure
                        image_paths.append(os.path.join(folder_save_path, file_name))
            elif os.path.isfile(path) and self._is_image_file(path):
                images.append(Image.open(path))
                image_paths.append(os.path.join(save_folder, os.path.basename(path)))

        # Run pipeline on images and save embeddings
        embeddings = self.run_on_images(images)

        for embedding, img_path in zip(embeddings, image_paths):
            # Save each embedding as a .npy file in the designated folder
            image_name = os.path.basename(img_path)
            embedding_save_path = os.path.join(os.path.dirname(img_path), f"{os.path.splitext(image_name)[0]}.npy")
            np.save(embedding_save_path, embedding)

        print(f"Embeddings saved in folder: {save_folder}")
        
        if save_unique_df:
            output_dict = {}
            for embedding, img_path in zip(embeddings, image_paths):
                # Save each embedding as a .npy file in the designated folder
                image_name = os.path.basename(img_path)
                output_dict[image_name] = embedding
            df_emb = pd.DataFrame.from_dict(output_dict, orient= 'columns')
            df_emb.to_csv("DAM_embeddings", index=False)
    

    @staticmethod
    def _is_image_file(file_path: str):
        """
        Checks if a file path corresponds to an image.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return os.path.splitext(file_path)[1].lower() in valid_extensions
