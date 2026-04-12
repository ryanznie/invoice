"""
Visualization helpers shared by tests and external clients.
"""

from PIL import Image, ImageDraw


def create_annotated_image(image, words, boxes, labels):
    """
    Create an annotated image with bounding boxes.

    Args:
        image: PIL Image
        words: List of words
        boxes: List of bounding boxes
        labels: List of predicted labels (may be shorter than words/boxes)

    Returns:
        Annotated PIL Image

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs are invalid
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

    if (
        not isinstance(words, list)
        or not isinstance(boxes, list)
        or not isinstance(labels, list)
    ):
        raise TypeError("words, boxes, and labels must be lists")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    num_to_annotate = min(len(words), len(boxes), len(labels))

    for i in range(num_to_annotate):
        box = boxes[i]
        label = labels[i]

        x0 = int(box[0] * img_width / 1000)
        y0 = int(box[1] * img_height / 1000)
        x1 = int(box[2] * img_width / 1000)
        y1 = int(box[3] * img_height / 1000)

        if (
            label == "HEURISTIC_MATCH"
            or label.startswith("LABEL_1")
            or label.startswith("LABEL_2")
        ):
            color = "red"
            width_box = 3
        else:
            color = "lightblue"
            width_box = 1

        draw.rectangle([x0, y0, x1, y1], outline=color, width=width_box)

    return img
