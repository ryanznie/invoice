"""
Base interface for invoice extraction models.

This module provides an abstract base class that all models must implement
for consistent benchmarking and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from PIL import Image
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """
    Standardized result format for all models.

    Attributes:
        invoice_number: Extracted invoice number (None if not found)
        confidence: Model confidence score (0-1, None if not applicable)
        method: Extraction method used ('heuristic', 'model', or 'hybrid')
        metadata: Additional model-specific information
    """

    invoice_number: Optional[str]
    confidence: Optional[float] = None
    method: str = "model"
    metadata: Optional[Dict[str, Any]] = None


class BaseInvoiceModel(ABC):
    """
    Abstract base class for invoice extraction models.

    All models should inherit from this class and implement the required methods.
    This ensures consistent interface for benchmarking different models.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.

        Args:
            model_config: Optional configuration dictionary containing model parameters
        """
        self.model_config = model_config or {}
        self.model_name = self.__class__.__name__

    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights and initialize all required components.

        This method should handle:
        - Loading model weights
        - Initializing processors/tokenizers
        - Setting device (CPU/GPU/MPS)
        - Any other model-specific setup

        Raises:
            Exception: If model loading fails
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Run inference on invoice data.

        Args:
            image: PIL Image of the invoice (optional for text-only models)
            words: List of OCR words (optional)
            boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000
            text: Plain text content (optional, alternative to words)
            **kwargs: Additional model-specific parameters

        Returns:
            InferenceResult containing prediction and metadata

        Raises:
            ValueError: If required inputs are missing
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration for logging/tracking.

        Returns:
            Dictionary containing:
                - model_name: Name/identifier of the model
                - model_version: Version string
                - checkpoint_path: Path to model checkpoint (if applicable)
                - device: Device used for inference
                - Any other relevant configuration
        """
        pass

    def cleanup(self) -> None:
        """
        Optional cleanup method for releasing resources.

        Override this if your model needs to explicitly release memory,
        close connections, or perform other cleanup tasks.
        """
        pass

    def __enter__(self):
        """Context manager support for automatic cleanup."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support for automatic cleanup."""
        self.cleanup()
