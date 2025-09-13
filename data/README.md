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

1.  The application will open in your web browser.
2.  You will see an image on the left and a text box with its content on the right.
3.  Edit the text in the text box to match the information in the image.
4.  Use the **Previous** and **Next** buttons to navigate through the image files.
5.  Click **Save and Next** to save your changes to `labels.json` and automatically move to the next image.
6.  Use the **Go to page** input to jump to a specific image by its number.

## Output

The corrected labels are saved in a file named `labels.json` in the `data/` directory. The file contains a JSON object where each key is the image filename and the value is the corrected text content.

**Example `labels.json`:**

```json
{
    "X00016469612.jpg": "<INVOICE#>",
    "X00016469613.jpg": "<INVOICE#>"
}
```
