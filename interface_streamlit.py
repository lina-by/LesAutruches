import streamlit as st
import base64
import os
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Dior Product Finder", layout="wide")
def add_background(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background("Capture d’écran 2025-01-30 à 03.18.15.png")

st.markdown(
    """
    <style>
    .main {
        padding-top: 5cm;  /* Appliquer une marge de 5 cm en haut pour tout le contenu */
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main" style="text-align: center;">
        <h1 style="color: #D2B48C;">Dior Product Finder</h1>
    </div>
    """, unsafe_allow_html=True
)

image_path = 'LogoCS.png'  
st.image(image_path, width=100)

DAM_DIR = "data/DAM"
TEST_DIR = "test_images_resized"
DAM_EMBEDDINGS_FILE = "dam_embeddings_FashionClip.pkl"
TEST_EMBEDDINGS_FILE = "test_embeddings_FashionClip.pkl"

def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

dam_embeddings = load_embeddings(DAM_EMBEDDINGS_FILE)
test_embeddings = load_embeddings(TEST_EMBEDDINGS_FILE)


def find_top_k_matches(test_embedding, dam_embeddings, k=10):
    similarities = []
    for dam_name, dam_embedding in dam_embeddings.items():
        similarity = np.dot(test_embedding, dam_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(dam_embedding)
        )
        similarities.append((dam_name, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).resize((256, 256))
    image_array = img_to_array(image)
    return preprocess_input(image_array)


test_image_names = list(test_embeddings.keys())
selected_test_image_name = st.selectbox("Choisissez une image test :", test_image_names)

if selected_test_image_name:
    test_image_path = os.path.join(TEST_DIR, selected_test_image_name)

    
    if os.path.exists(test_image_path):
        
        st.markdown(
            """
            <div style="text-align: center;">
            """, unsafe_allow_html=True)
        st.image(test_image_path, caption="Image de test", width=250)  # Taille de l'image ajustée à 250 px de large
        st.markdown("</div>", unsafe_allow_html=True) 
    else:
        st.error(f"L'image '{selected_test_image_name}' n'a pas été trouvée dans le répertoire {TEST_DIR}.")

    test_embedding = test_embeddings[selected_test_image_name]

    k = st.slider("Nombre de correspondances à afficher :", min_value=1, max_value=10, value=5, step=1)


    top_matches = find_top_k_matches(test_embedding, dam_embeddings, k=k)

    st.subheader(f"Top {k} correspondances trouvées :")
    selected_match = None

    for i, (dam_name, score) in enumerate(top_matches):
        dam_image_path = os.path.join(DAM_DIR, dam_name)
       
        if os.path.exists(dam_image_path):
            
            st.image(dam_image_path, caption=f"{dam_name}\nScore: {score:.4f}", width=400)
        else:
            st.error(f"L'image de correspondance '{dam_name}' n'a pas été trouvée dans le répertoire {DAM_DIR}.")

        dam_name_reference = dam_name.replace(".jpeg", "")  # Supprimer l'extension .jpeg

        
        st.markdown(
            f"""
            <h2 style="font-size: 30px; text-align: center;">Référence : {dam_name_reference}</h2>
            """, unsafe_allow_html=True
        )




    

    