# Data Labeling

The Streamlit app lets you review SROIE2019 invoice images and edit their labels.

## Prepare and run

Place the dataset under data/SROIE2019 with train/img, train/box, test/img, and test/box. Install the development extra from the repository root, then launch the app from data/:

    cd data
    source ../.venv/bin/activate
    streamlit run app.py

Choose Train or Test in the sidebar, select All or Ambiguous, edit the label, and use Save and Next. The app reads JPG images and matching TXT OCR files.

The app writes labels.json or test_labels.json in the data directory. Edits that replace an ambiguous label are appended to ambiguous_edits.log there.

The preprocessing script converts OCR files and labels into model-ready JSON:

    uv run python scripts/preprocess.py --help

See [Developer Setup](../docs/DEV_SETUP.md) for environment setup and [dataset and labeling notes (Notion)](https://www.notion.so/Dataset-Documentation-Notes-1609faffd568479dbaf1c072b23c472d) for labeling context and heuristics.
