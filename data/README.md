# Invoice/Receipt Data Labeling Tool

This Streamlit application is a tool for labeling and correcting text extracted from invoice and receipt images.

## Features

-   Displays an image and its corresponding OCR-extracted text side-by-side.
-   Allows users to edit and correct the text.
-   Saves the corrected labels to a `labels.json` file.
-   Provides easy navigation between images.

## Directory Structure

Before running the application, ensure your data is organized in the following structure within the `data/` directory:

```
SROIE2019/
└── train/
    ├── img/      # Contains .jpg images
    │   ├── X00016469612.jpg
    │   └── ...
    └── box/      # Contains .txt files with extracted text
        ├── X00016469612.txt
        └── ...
```

You will need to download the [data](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2?resource=download) and unpack in `data/`.

## How to Run

1.  **Install dependencies:**

    ```bash
    uv pip install streamlit pillow
    ```

2.  **Run the Streamlit app:**

    Navigate to the `data/` directory and run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

## How to Use

### Basic Navigation

1.  The application will open in your web browser.
2.  You will see an image on the left and a text box with its content on the right.
3.  The current file name and position (e.g., "X00016469612.jpg (1/626)") is displayed at the top.

### Editing Labels

1.  **Review the image**: Look at the invoice/receipt image on the left side.
2.  **Edit the text**: In the text area on the right, enter the invoice number or ID from the image.
    -   For simple invoice numbers, just enter the number (e.g., `7030F715`)
    -   For complex multi-part IDs, include all parts (e.g., `CS-SA-0096677` or `18124/102/T0146`)
    -   If the invoice number is unclear or ambiguous, enter `ambiguous`. The change is automatically logged to `ambiguous_edits.log` with a timestamp.

3.  **Save your work**: Click **💾 Save and Next** to save your label and automatically advance to the next image.

### Navigation Controls

-   **⬅️ Previous**: Go to the previous image
-   **Next ➡️**: Go to the next image without saving
-   **💾 Save and Next**: Save the current label and move to the next image
-   **Go to page**: Enter a page number (1-based) to jump directly to that image
-   **Go to file**: Enter a specific filename (e.g., `X00016469612.jpg`) and click **Go** to jump to that file

### Mode Selection (Sidebar)

The sidebar provides two important filters:

1.  **Dataset Mode**:
    -   **Train**: Work with training data (`SROIE2019/train/`)
    -   **Test**: Work with test data (`SROIE2019/test/`)

2.  **Filter by Label**:
    -   **All**: Show all images in the dataset
    -   **Ambiguous**: Show only images previously labeled as "ambiguous" for review

## Output

The corrected labels are saved in JSON files in the `data/` directory:

-   **`labels.json`**: Contains labels for training data
-   **`test_labels.json`**: Contains labels for test data
-   **`ambiguous_edits.log`**: Logs all edits made to previously ambiguous labels

Each JSON file contains a mapping from image filename to the invoice number/ID.

**Example `labels.json`:**

```json
{
    "X00016469612.jpg": "7030F715",
    "X00016469613.jpg": "CS-SA-0096677",
    "X00016469614.jpg": "ambiguous"
}
```

## Data

### Dataset Documentation
Documentation on the dataset can be found in this [Notion document](https://www.notion.so/Dataset-Documentation-Notes-1609faffd568479dbaf1c072b23c472d). Labeled data (using this tool) was released on [Kaggle](https://www.kaggle.com/datasets/ryanznie/sroie-datasetv2-with-labels) and [HuggingFace](https://huggingface.co/datasets/ryanznie/SROIE_2019_with_labels).

### Labeling Heuristics

For detailed labeling instructions and examples, please refer to the [documentation on Notion](https://www.notion.so/Heuristics-Details-53af761344b7402fac834031244e032a#27ac697d927c8087ab97ebfbb0d23a38).

