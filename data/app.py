import streamlit as st
import os
import json
from PIL import Image
import datetime

# --- Configuration ---
AMBIGUOUS_LOG_FILE = "ambiguous_edits.log"

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Invoice Data Labeling and Navigation Tool")

# --- Mode Selection ---
mode = st.sidebar.radio("Select Mode", ("Train", "Test"))

if mode == "Train":
    IMG_DIR = os.path.join("SROIE2019", "train", "img")
    BOX_DIR = os.path.join("SROIE2019", "train", "box")
    LABELS_FILE = "labels.json"
else:
    IMG_DIR = os.path.join("SROIE2019", "test", "img")
    BOX_DIR = os.path.join("SROIE2019", "test", "box")
    LABELS_FILE = "test_labels.json"

sub_mode = st.sidebar.radio("Filter by Label", ("All", "Ambiguous"))

st.sidebar.write(f"**Dataset:** `{IMG_DIR}`")
st.sidebar.write(f"**Labels:** `{LABELS_FILE}`")


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


def load_labels():
    """Load labels from the JSON file."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def load_data(img_file, box_file):
    """Load image and box content."""
    img_path = os.path.join(IMG_DIR, img_file)
    box_path = os.path.join(BOX_DIR, box_file)

    image = Image.open(img_path)

    try:
        with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
            box_content = f.read()
    except FileNotFoundError:
        box_content = ""
    return image, box_content


def save_label(img_filename, text_content):
    """Save a label to the JSON file and log edits to ambiguous files."""
    labels = load_labels()
    old_label = labels.get(img_filename, "").strip().lower()

    # Log if an 'ambiguous' file is changed to something else
    if old_label == "ambiguous" and text_content.strip().lower() != "ambiguous":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - Edited ambiguous file: {img_filename}. New label: '{text_content.strip()}'\n"
        with open(AMBIGUOUS_LOG_FILE, "a") as log_file:
            log_file.write(log_message)

    labels[img_filename] = text_content
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=4)


img_files, box_files = get_file_paths()

# --- Filtering Logic ---
if sub_mode == "Ambiguous":
    labels = load_labels()
    ambiguous_files = [
        f for f, label in labels.items() if label.strip().lower() == "ambiguous"
    ]

    # Filter img_files and box_files to only include ambiguous ones
    img_files = [f for f in img_files if f in ambiguous_files]
    box_files = [os.path.splitext(f)[0] + ".txt" for f in img_files]

if not img_files:
    st.warning("No images found. Please check your data directories.")
    st.stop()

# --- Session State ---
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Reset index if it's out of bounds after filtering
if st.session_state.current_index >= len(img_files):
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
        st.image(image, width="stretch")

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
