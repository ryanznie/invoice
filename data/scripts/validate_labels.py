import logging
import string
import json
import argparse
from tqdm import tqdm


def validate_label(label_content, img_filename, labels_json_path):
    """Validates a single label for an image."""
    warnings_count = 0
    if label_content is None:
        logging.warning(f"⚠️ Missing JSON label for: {img_filename}")
        return (
            None,
            True,
            1,
        )  # Return None, missing status, and 1 for the missing warning

    if not label_content and isinstance(label_content, str):
        logging.warning(f"⚠️ Empty JSON label for: {img_filename}")
        return (
            label_content,
            False,
            1,
        )  # Return the empty label, not missing, and 1 for the warning

    if isinstance(label_content, str):
        original_label = label_content
        # Check for whitespace
        if label_content.strip() != label_content:
            logging.warning(
                f"⚠️ Label '{original_label}' in {img_filename} has leading/trailing whitespace. "
                f"Will need to fix in {labels_json_path}!"
            )
            label_content = label_content.strip()  # Re-assign the stripped string
            warnings_count += 1

        # Check for punctuation
        if label_content and label_content[0] in string.punctuation:
            logging.warning(
                f"⚠️ Label '{label_content}' in {img_filename} starts with punctuation."
            )
            warnings_count += 1
        if label_content and label_content[-1] in string.punctuation:
            logging.warning(
                f"⚠️ Label '{label_content}' in {img_filename} ends with punctuation."
            )
            warnings_count += 1

    return (
        label_content,
        False,
        warnings_count,
    )  # Return the validated label, missing status, and warning count


def main(json_path):
    """Loads a JSON file and validates its labels."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: The file '{json_path}' is not a valid JSON file.")
        return

    missing_labels = 0
    total_warnings = 0
    total_labels = len(labels_data)

    for img_filename, label_content in tqdm(
        labels_data.items(), desc=f"Validating labels in {json_path}"
    ):
        _, is_missing, warnings_count = validate_label(
            label_content, img_filename, json_path
        )
        total_warnings += warnings_count
        if is_missing:
            missing_labels += 1

    logging.info(f"===== VALIDATION RESULTS FOR {json_path} =====")
    logging.info(f"Total labels: {total_labels}")
    if total_warnings > 0:
        logging.warning(f"⚠️ Total warnings: {total_warnings}")
    else:
        logging.info("No warnings found.")

    if missing_labels > 0:
        logging.warning(f"⚠️ Missing labels: {missing_labels}")
    else:
        logging.info("No missing labels found.")
    logging.info("Validation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate labels in a JSON file.")
    parser.add_argument("json_path", help="The path to the JSON file to validate.")
    args = parser.parse_args()
    main(args.json_path)
