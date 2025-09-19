import streamlit as st
import os
import json
from PIL import Image

# --- Configuration ---
IMG_DIR = os.path.join("SROIE2019", "train", "img")
BOX_DIR = os.path.join("SROIE2019", "train", "box")
LABELS_FILE = "labels.json"


# --- Helper Functions ---
def get_file_paths():
    """Get sorted lists of image and box file paths."""
    try:
        img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
        box_files = [os.path.splitext(f)[0] + ".txt" for f in img_files]
        return img_files, box_files
    except FileNotFoundError:
        st.error(f"Directory not found. Make sure '{IMG_DIR}' and '{BOX_DIR}' exist.")
        return [], []


def load_data(img_file, box_file):
    """Load image and box content."""
    img_path = os.path.join(IMG_DIR, img_file)
    box_path = os.path.join(BOX_DIR, box_file)

    image = Image.open(img_path)

    try:
        with open(box_path, "r", encoding="utf-8") as f:
            box_content = f.read()
    except FileNotFoundError:
        box_content = ""
    return image, box_content


def save_label(img_filename, text_content):
    """Save a label to the JSON file."""
    data = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass  # Overwrite if file is corrupted

    data[img_filename] = text_content

    with open(LABELS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Invoice / Receipt Data Labeling Tool")

img_files, box_files = get_file_paths()

if not img_files:
    st.warning("No images found. Please check your data directories.")
    st.stop()

# --- Session State ---
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# --- Navigation and Display ---
index = st.session_state.current_index

# Update index for display
index = st.session_state.current_index
img_file = img_files[index]
box_file = box_files[index]

# --- Layout ---
# Define containers
header = st.container()
content = st.container()
footer = st.container()

with header:
    st.write(f"**File:** {img_file} ({index + 1}/{len(img_files)})")
    # Navigation and Save buttons
    col1, col2, col3 = st.columns(3)
    prev_button = col1.button("⬅️ Previous")
    next_button = col2.button("Next ➡️")
    save_button = col3.button("💾 Save and Next")

# Get the text from the text area first
with content:
    img_col, text_col = st.columns([2, 3])
    image, box_content = load_data(img_file, box_file)

    with img_col:
        st.image(image, use_container_width=True)

    with text_col:
        edited_text = st.text_area("Label Text", value=box_content, height=500)

# --- Button Logic ---
if prev_button:
    if index > 0:
        st.session_state.current_index -= 1
        st.rerun()

if next_button:
    if index < len(img_files) - 1:
        st.session_state.current_index += 1
        st.rerun()

if save_button:
    save_label(img_file, edited_text)
    st.success(f"Saved label for {img_file}")
    if index < len(img_files) - 1:
        st.session_state.current_index += 1
        st.rerun()

# Footer with Go To Page and Go To File
with footer:
    st.write("___")
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        target_page = st.number_input(
            "Go to page:",
            min_value=1,
            max_value=len(img_files),
            value=index + 1,
            step=1,
            label_visibility="collapsed",
        )
        if target_page != index + 1:
            st.session_state.current_index = target_page - 1
            st.rerun()

    with col2:
        go_to_filename = st.text_input(
            "Go to file:", placeholder="Enter filename...", label_visibility="collapsed"
        )

    with col3:
        if st.button("Go"):
            if go_to_filename in img_files:
                st.session_state.current_index = img_files.index(go_to_filename)
                st.rerun()
            elif go_to_filename:
                st.error(f"File '{go_to_filename}' not found.")
