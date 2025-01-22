import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
from utils import find_top_k_matches
from vectorizers.efficient_net import EfficientNet

# Directories
REFERENCE_DIR = r"data\DAM"
EMBEDDINGS_DIR = r'embeddings\efficient_net_no_preprocessing\DAM'
TEST_DIR = r"data\test_image_headmind"
CSV_FILE = "results.csv"
model = EfficientNet()

# Load reference embeddings
@st.cache_resource
def load_reference_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            embeddings[name] = np.load(os.path.join(EMBEDDINGS_DIR, file))
    return embeddings

# Load test images
def get_unprocessed_images():
    processed_images = set()
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        processed_images.update(df["image"].tolist())
    
    return [f for f in os.listdir(TEST_DIR) if f not in processed_images and f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Load reference image
def get_image(file_name):
    for ext in [".jpg", ".png", ".jpeg"]:
        match_image_path = os.path.join(REFERENCE_DIR, file_name + ext)
        if os.path.exists(match_image_path):
            return Image.open(match_image_path).resize((150, 150))
    return None  # Return None if image is not found

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "selected_matches" not in st.session_state:
    st.session_state.selected_matches = []

st.title("Image Matching")

reference_embeddings = load_reference_embeddings()
unprocessed_images = get_unprocessed_images()

# Check if there are any images to process
if st.session_state.index >= len(unprocessed_images):
    st.write("✅ All images have been processed!")
    if st.button("Save Results"):
        df = pd.DataFrame(st.session_state.selected_matches)
        if os.path.exists(CSV_FILE):
            df_existing = pd.read_csv(CSV_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        st.success("Selections saved!")
    st.stop()

test_image_name = unprocessed_images[st.session_state.index]
test_image_path = os.path.join(TEST_DIR, test_image_name)
test_image = Image.open(test_image_path)

st.image(test_image, caption="Test Image", use_container_width=True)

# Extract features and find matches
test_embedding = model(test_image)
matches = find_top_k_matches(test_embedding, reference_embeddings)

options = []
match_images = []
for match_name, _ in matches:
    match_image = get_image(match_name)
    if match_image:
        match_images.append((match_name, match_image))
        options.append(match_name)
options.append("Not in list")

# Display images in a grid format (2 rows, 5 columns)
cols = st.columns(5)
for idx, (name, img) in enumerate(match_images):
    with cols[idx % 5]:
        st.image(img, caption=name, use_container_width=True)

selected_match = st.selectbox(
    f"Select the correct match for {test_image_name}:",
    options,
    key=test_image_name
)

if st.button("Next Image"):
    st.session_state.selected_matches.append({"image": test_image_name, "match": selected_match})
    st.session_state.index += 1
    st.rerun()
