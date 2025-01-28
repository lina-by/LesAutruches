# fashion_clip_embeddings.py

# --- Importations ---
import os
import zipfile
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from transformers import CLIPProcessor, CLIPModel


# --- Chargement du modèle et du processeur ---
def load_model_and_processor():
    """
    Charge le modèle Fashion-CLIP et le processeur à partir de Hugging Face.
    
    Returns:
        tuple: (model, processor)
    """
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    return model, processor


# --- Extraction des embeddings ---
def extract_embeddings(image_data, model, processor):
    """
    Extrait les embeddings pour une collection d'images.
    
    Parameters:
        image_data (dict): Dictionnaire contenant {nom_image: chemin ou PIL.Image}.
        model: Modèle Fashion-CLIP.
        processor: Processor Fashion-CLIP.
        
    Returns:
        dict: {nom_image: embedding}.
    """
    embeddings = {}
    for img_name, img_path in tqdm(image_data.items(), desc="Calcul des embeddings"):
        
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings[img_name] = image_embedding.squeeze().numpy()  # Convertir en numpy array
    return embeddings


# --- Fonction de recherche d'image la plus proche ---
def find_closest_image(test_embedding, dam_embeddings):
    """
    Trouve l'image la plus proche dans DAM en termes de similarité de cosinus.
    
    Parameters:
        test_embedding (numpy array): Embedding de l'image de test.
        dam_embeddings (dict): Dictionnaire contenant {nom_image: embedding}.
    
    Returns:
        tuple: (nom_image, score_de_similarité)
    """
    closest_image = None
    highest_similarity = -1  # Similarité de cosinus max
    
    for dam_name, dam_embedding in dam_embeddings.items():
        similarity = 1 - cosine(test_embedding, dam_embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_image = dam_name
    
    return closest_image, highest_similarity


# --- Visualisation des images les plus proches ---
def visualize_closest_image(test_image_path, closest_image_path):
    """
    Affiche l'image de test et l'image correspondante côte à côte.
    
    Parameters:
        test_image_path (str): Chemin de l'image de test.
        closest_image_path (str): Chemin de l'image correspondante dans DAM.
    """
    test_image = Image.open(test_image_path)
    closest_image = Image.open(closest_image_path)
    

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title("Image de Test")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(closest_image)
    plt.title("Image la Plus Proche (DAM)")
    plt.axis("off")
    
    plt.show()



def extract_zip(zip_path, extract_path):
    """
    Extrait le contenu d'un fichier ZIP vers un dossier spécifié.
    
    Parameters:
        zip_path (str): Chemin vers le fichier ZIP.
        extract_path (str): Dossier où les fichiers seront extraits.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Fichiers extraits dans {extract_path}")



def main(zip_path, extract_path):
    
    model, processor = load_model_and_processor()

    extract_zip(zip_path, extract_path)
    
    dam_folder = os.path.join(extract_path, "Rotation/DAM")
    test_folder = os.path.join(extract_path, "Rotation/test_images_headmind")
    
 
    dam_images = {img: os.path.join(dam_folder, img) for img in os.listdir(dam_folder) if img.endswith((".png", ".jpg", ".jpeg"))}
    test_images = {img: os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.endswith((".png", ".jpg", ".jpeg"))}
    dam_embeddings = extract_embeddings(dam_images, model, processor)
    test_embeddings = extract_embeddings(test_images, model, processor)
    

    for test_image_name in tqdm(test_embeddings.keys(), desc="Traitement des images de test"):
      
        test_image_embedding = test_embeddings[test_image_name]

        closest_image_name, similarity_score = find_closest_image(test_image_embedding, dam_embeddings)
        print(f"L'image de test '{test_image_name}' est la plus proche de '{closest_image_name}' avec une similarité de {similarity_score:.2f}")

        test_image_path = os.path.join(test_folder, test_image_name)
        closest_image_path = os.path.join(dam_folder, closest_image_name)

        visualize_closest_image(test_image_path, closest_image_path)



if __name__ == "__main__":
    zip_path = "/content/Rotation.zip"  
    extract_path = "/content/rotation_extracted"  
    
    main(zip_path, extract_path)
