import numpy as np
import os
from transformers import (
    AutoImageProcessor, 
    SegformerForSemanticSegmentation, 
    AutoModelForImageSegmentation, 
    ViTImageProcessor, 
    ViTModel, 
    pipeline
)
from PIL import Image
import torch
from tqdm import tqdm

# Initialize the segmentation pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Initialize the ViT model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

def extract_features(image_path, output_folder=None, overwrite=False):
    """
    Extract features from an image using a segmentation pipeline and a ViT model.
    Optionally save the features to a file.
    """
    image_name = os.path.basename(image_path)
    if (output_folder is not None
        and not overwrite
        and image_name in os.listdir(output_folder)):
        return "already saved"

    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply segmentation to extract the object of interest
    segmented_result = pipe(image).convert("RGB")
    
    # Pass through the ViT model
    inputs = processor(images=segmented_result, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the last hidden states (features)
    last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    
    # Optionally save the features to a file
    if output_folder is not None:
        feature_save_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.pt")
        torch.save(last_hidden_states, feature_save_path)
    
    return last_hidden_states


if __name__ == "__main__":
    # Directory containing images
    image_directory = r"data/DAM"
    output_directory = r"data/feature_output"
    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists
    
    for image_name in tqdm(os.listdir(image_directory)):
        try:
            image_path = os.path.join(image_directory, image_name)
            features = extract_features(image_path, output_folder=output_directory)
        except Exception as e:
            print('skip', e, image_name)
