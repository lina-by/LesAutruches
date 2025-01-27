import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
from utils import find_top_k_matches
from vectorizers.efficient_net import EfficientNet
from image_preprocessing.rotations import OrientationImages

# Directories
REFERENCE_DIR = r"data\DAM"
EMBEDDINGS_DIR = r'embeddings\efficient_net_no_preprocessing\DAM'
TEST_DIR = r"data\test_image_headmind"
CSV_FILE = "results.csv"

# Initialize model
model = EfficientNet()
rotation = OrientationImages()

@st.cache_resource
def load_reference_embeddings():
    """Load reference embeddings."""
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            embeddings[name] = np.load(os.path.join(EMBEDDINGS_DIR, file))
    return embeddings

def get_unprocessed_images():
    """Return a list of unprocessed test images."""
    processed_images = set()
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        processed_images.update(df["image"].tolist())
    return [f for f in os.listdir(TEST_DIR) if f not in processed_images and f.lower().endswith(('png', 'jpg', 'jpeg'))]

def load_image(image_path):
    """Fast image loading using OpenCV."""
    image = Image.open(image_path)  # Load image as NumPy array
    image = rotation(image).resize((224, 224))
    return image

@st.cache_resource
def extract_features(image_path):
    """Feature extraction with OpenCV."""
    image = load_image(image_path)
    embedding = model(image)
    return embedding

def get_image(file_name):
    """Retrieve a reference image."""
    for ext in [".jpg", ".png", ".jpeg"]:
        match_image_path = os.path.join(REFERENCE_DIR, file_name + ext)
        if os.path.exists(match_image_path):
            return Image.open(match_image_path).resize((150, 150))
    return None

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "selected_matches" not in st.session_state:
    st.session_state.selected_matches = []

st.title("🔍 Image Matching")

# Load data
reference_embeddings = load_reference_embeddings()
unprocessed_images = get_unprocessed_images()

# Save function
def save_results():
    """Save results to CSV."""
    if st.session_state.selected_matches:
        df = pd.DataFrame(st.session_state.selected_matches)
        if os.path.exists(CSV_FILE):
            df_existing = pd.read_csv(CSV_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        st.success("✅ Selections saved successfully!")

# Save button (always visible)
if st.button("💾 Save Results"):
    save_results()

# Check if all images are processed
if st.session_state.index >= len(unprocessed_images):
    st.write("✅ **All images have been processed!**")
    st.stop()

# Process next image
test_image_name = unprocessed_images[st.session_state.index]
test_image_path = os.path.join(TEST_DIR, test_image_name)

st.write(f"📷 **Processing Image:** `{test_image_name}`")
test_image = Image.open(test_image_path)
st.image(test_image, caption="Test Image", use_container_width=True)

# Feature extraction
test_embedding = extract_features(test_image_path)

# Get matches
matches = find_top_k_matches(test_embedding, reference_embeddings)

# Display clickable images in a 2-row, 5-column grid
cols = st.columns(5)
selected_match = None

for idx, (match_name, _) in enumerate(matches):
    match_image = get_image(match_name)
    if match_image:
        with cols[idx % 5]:
            if st.button(f"img_{match_name}", key=f"btn_{match_name}"):
                selected_match = match_name
            st.image(match_image, caption=match_name, use_container_width=True)

# "Not in list" button
if st.button("Not in list", key="not_in_list"):
    selected_match = "Not in list"

# Move to next image when a match is selected
if selected_match:
    st.session_state.selected_matches.append({"image": test_image_name, "match": selected_match})
    st.session_state.index += 1
    st.rerun()
