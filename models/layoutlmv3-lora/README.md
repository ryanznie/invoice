# LayoutLMv3-LoRA for Invoice Number Extraction

## Model Summary

| Field | Details |
|-------|----------|
| **Base Model** | microsoft/layoutlmv3-base |
| **Model Name** | layoutlmv3-lora-invoice-number |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) |
| **Task** | Token Classification — Invoice Number Extraction |
| **Dataset** | SROIE 2019 (invoice subset) |
| **License** | MIT (inherited from base model) |
| **Developed by** | Ryan Z. Nie |

---

## Model Description

This model fine-tunes **LayoutLMv3-base** using **LoRA** for the task of **invoice number extraction** from scanned receipts and invoices. It leverages both visual (layout) and textual information from documents to identify and extract invoice numbers accurately.

The model is lightweight and memory-efficient, trained with low-rank adapters on attention and MLP layers to minimize computational and storage costs without sacrificing accuracy.

---

## Intended Use

### Primary Task
Token classification for invoice number extraction from document images.

### Input
OCR-parsed document images containing:
- Text words
- Bounding boxes
- Layout information

### Output
Invoice number tokens tagged using BIO labels.

### Example Use Case
Extracting invoice or bill numbers from scanned receipts in accounting automation systems or document understanding pipelines.

---

## Example Usage

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
import torch

# Load processor and model
processor = AutoProcessor.from_pretrained("ryanznie/layoutlmv3-lora-invoice-number")
model = AutoModelForTokenClassification.from_pretrained("ryanznie/layoutlmv3-lora-invoice-number")

# Example input
image = Image.open("invoice_sample.jpg")
words = ["Invoice", "No.", "PEGIV-1030765"]
boxes = [[100, 200, 200, 230], [210, 200, 250, 230], [260, 200, 400, 230]]

# Preprocess
encoding = processor(image, words, boxes=boxes, return_tensors="pt")

# Predict
outputs = model(**encoding)
predictions = torch.argmax(outputs.logits, dim=-1)

# Print results
print(predictions)
```

---

## Training Details

### Dataset
[SROIE 2019 w/ invoices Dataset](https://www.kaggle.com/datasets/ryanznie/sroie-datasetv2-with-labels)
[Dataset Documentations](https://www.notion.so/Dataset-Documentation-Notes-1609faffd568479dbaf1c072b23c472d)

### Training Configuration
- **Hardware:** Apple MacBook M2 (8-core CPU, 16GB RAM)
- **Acceleration:** Apple Metal (MPS)
- **Duration:** ~1.5–2 hours
- **Framework:** Hugging Face Transformers
- **Fine-tuning Method:** LoRA (on attention and MLP layers)
- **Optimization Objective:** FocalLoss
- **Training Mode:** Mixed-precision training

### Technical Specifications
- **Base Architecture:** LayoutLMv3
- **Adapter Type:** LoRA
- **Target Modules:** Attention and MLP layers
- **Objective:** Token classification for invoice number extraction

### Framework Versions

| Component | Version |
|------------|----------|
| Python | 3.11.13 |
| PyTorch | 2.8.0 |
| Transformers | 4.57.0 |
| PEFT | 0.17.1 |

---

## Performance

The model performs well on invoice number extraction tasks, correctly combining multi-token predictions into complete invoice numbers (e.g., `PEGIV-1030765`). After postprocessing, it achieves ~81% accuracy on the SROIE 2019 test set.

### Evaluation Metrics
- F1-score for entity-level invoice number recognition
- Precision and recall measured on validation split of SROIE 2019
- Overall accuracy

---

## Limitations

- The model is specialized for **English-language invoices** and **SROIE-like layouts**
- May mispredict when invoice number patterns differ significantly (e.g., multiple dashes or alphanumeric codes not seen during training)
- Performance may degrade on handwritten or low-quality scans
- Limited to document types similar to those in the training dataset

---

## Ethical Considerations

- Ensure document data used respects privacy and does not contain sensitive or personal information
- The model should not be used to process private or confidential documents without explicit consent
- Consider data protection regulations (GDPR, CCPA, etc.) when processing invoices
- Verify accuracy before using in production systems that affect financial decisions

---

## Environmental Impact

Carbon emissions estimated using the [Machine Learning Impact Calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Apple MacBook M2 (8-core)
- **Hours Used:** ~2 hours
- **Cloud Provider:** Local (no cloud compute)
- **Compute Region:** United States
- **Carbon Emitted:** Negligible (< 0.01 kg CO₂e)

---

## Glossary

- **LayoutLMv3** — A transformer-based model for document understanding that fuses text, layout, and image embeddings
- **LoRA (Low-Rank Adaptation)** — A lightweight fine-tuning method where small trainable matrices are added to specific layers (e.g., attention and MLP), enabling efficient adaptation without updating full model weights
- **Token Classification** — A form of sequence labeling where each token is assigned a class label, used here for identifying invoice numbers within document text
- **BIO Labels** — Begin, Inside, Outside tagging scheme for named entity recognition

---

## Citation

If you use this model, please cite:

```bibtex
@misc{nie2025layoutlmv3lora,
  author = {Ryan Z. Nie},
  title = {LayoutLMv3-LoRA for Invoice Number Extraction},
  year = {2025},
  howpublished = {\url{https://huggingface.co/ryanznie/layoutlmv3-lora-invoice-number}},
  note = {Fine-tuned LayoutLMv3 using LoRA on the SROIE 2019 dataset.}
}
```

---

## Contact Information

For feedback, questions, or collaboration inquiries:

- **Hugging Face:** [@ryanznie](https://huggingface.co/ryanznie)
- **Email:** [ryanznie [at] gatech [dot] edu](mailto:ryanznie@gatech.edu)
- **GitHub:** [@ryanznie](https://github.com/ryanznie)
- **LinkedIn:** [in/ryanznie](https://www.linkedin.com/in/ryanznie/)

**Model Card Author:** Ryan Z. Nie \
**Date:** October 2025