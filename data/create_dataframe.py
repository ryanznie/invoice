import os
import pandas as pd
import json
import logging
import string
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define paths relative to the project root
base_path = "data/SROIE2019/train"
img_dir = os.path.join(base_path, "img")
box_dir = os.path.join(base_path, "box")
entities_dir = os.path.join(base_path, "entities")
labels_json_path = "data/labels.json"

# Load labels from JSON
with open(labels_json_path, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

# Get the list of file basenames from the 'box' directory
box_files = os.listdir(box_dir)
file_basenames = [os.path.splitext(f)[0] for f in box_files if f.endswith(".txt")]

data = []
missing_labels_count = 0

for basename in tqdm(file_basenames, desc="Processing files"):
    img_filename = f"{basename}.jpg"
    box_filepath = os.path.join(box_dir, f"{basename}.txt")
    entities_filepath = os.path.join(entities_dir, f"{basename}.txt")

    # --- Validation: Check for missing files ---
    img_path = os.path.join(img_dir, img_filename)
    if not os.path.exists(img_path):
        logging.warning(f"Missing image file: {img_filename}")

    if os.path.exists(box_filepath) and os.path.exists(entities_filepath):

        # Read content from the entities file
        with open(entities_filepath, "r", encoding="utf-8") as f:
            entity_content = f.read()

        # Get label from JSON data
        label_content = labels_data.get(img_filename)

        # --- Validation: Check for missing JSON labels ---
        if label_content is None:
            logging.warning(f"Missing JSON label for: {img_filename}")
            missing_labels_count += 1
        elif isinstance(label_content, str):
            # --- Validation: Check label for whitespace and punctuation ---
            original_label = label_content
            # Check for whitespace
            if label_content.strip() != label_content:
                logging.warning(
                    f"Label '{original_label}' in {img_filename} has leading/trailing whitespace. Fixed in dataframe. Will need to fix in labels.json!"
                )
                label_content = label_content.strip()  # Re-assign the stripped string

            # Check for punctuation
            if label_content and label_content[0] in string.punctuation:
                logging.warning(
                    f"Label '{label_content}' in {img_filename} starts with punctuation."
                )
            if label_content and label_content[-1] in string.punctuation:
                logging.warning(
                    f"Label '{label_content}' in {img_filename} ends with punctuation."
                )

        # Read and process each line from the box file
        with open(box_filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    coords = parts[:8]
                    text = ",".join(parts[8:])

                    data.append(
                        {
                            "file": img_filename,
                            "x0": int(coords[0]),
                            "y0": int(coords[1]),
                            "x1": int(coords[2]),
                            "y1": int(coords[3]),
                            "x2": int(coords[4]),
                            "y2": int(coords[5]),
                            "x3": int(coords[6]),
                            "y3": int(coords[7]),
                            "text": text,
                            "entities": entity_content,
                            "labels": label_content,
                        }
                    )

# Log the total count of missing labels
if missing_labels_count > 0:
    total_files = len(file_basenames)
    percentage = (missing_labels_count / total_files * 100) if total_files > 0 else 0
    logging.info(
        f"Total number of missing labels: {missing_labels_count} out of {total_files} files ({percentage:.2f}%)."
    )

# Create a pandas DataFrame
df = pd.DataFrame(data)

# --- Validation: Check for duplicates ---
if df.duplicated().any():
    duplicate_rows = df[df.duplicated(keep=False)]
    logging.warning(f"Found {len(duplicate_rows)} duplicate rows.")

## TODO: validating dataset, check for duplicates, missing imgs, cleaning up labels (\n), trimming

# Save the DataFrame to a CSV file
output_path = "data/test_invoices.csv"
df.to_csv(output_path, index=False)

print(f"DataFrame saved to {output_path}")
print(f"\nDataFrame shape: {df.shape}")
