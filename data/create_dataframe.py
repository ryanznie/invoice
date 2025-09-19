import os
import pandas as pd
import json

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

for basename in file_basenames:
    img_filename = f"{basename}.jpg"
    box_filepath = os.path.join(box_dir, f"{basename}.txt")
    entities_filepath = os.path.join(entities_dir, f"{basename}.txt")

    # Check if all corresponding files exist
    if (
        os.path.exists(os.path.join(img_dir, img_filename))
        and os.path.exists(box_filepath)
        and os.path.exists(entities_filepath)
    ):

        # Read content from the entities file
        with open(entities_filepath, "r", encoding="utf-8") as f:
            entity_content = f.read()

        # Get label from JSON data
        label_content = labels_data.get(img_filename)

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

# Create a pandas DataFrame
df = pd.DataFrame(data)

## TODO: validating dataset, check for duplicates, missing imgs, cleaning up labels (\n), trimming

# Save the DataFrame to a CSV file
output_path = "data/invoices.csv"
df.to_csv(output_path, index=False)

print(f"DataFrame saved to {output_path}")
print(f"\nDataFrame shape: {df.shape}")
