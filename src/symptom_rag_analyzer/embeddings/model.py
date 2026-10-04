"""Biomedical embedding model using Sentence Transformers."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class BiomedicalEmbeddingModel:
    """
    Generates embeddings for biomedical text using Sentence Transformers.

    Uses the BioBERT model trained on multiple biomedical and general NLI datasets
    to produce 768-dimensional embeddings suitable for semantic search and retrieval.

    Attributes:
        model_name: The name of the Sentence Transformer model to use.
        model: The loaded Sentence Transformer model instance.
        embedding_dim: The dimensionality of the output embeddings (768).
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    ):
        """
        Initialize the biomedical embedding model.

        The model is loaded once during initialization to avoid repeated loading
        on each embedding call. The default model is trained on biomedical and
        general NLI datasets (mnli-snli-scinli-scitail-mednli-stsb).

        Args:
            model_name: The Sentence Transformer model identifier.
                Default is the BioBERT model specified in the base paper.

        Raises:
            ValueError: If the model cannot be loaded.
            RuntimeError: If model initialization fails.
        """
        self.model_name = model_name

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Sentence Transformer model '{model_name}': {e}"
            ) from e

        # Verify embedding dimension matches expected output
        self.embedding_dim = 768

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A 1D numpy array of shape (768,) containing the embedding vector.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text)}")

        if not text.strip():
            raise ValueError("text cannot be empty or whitespace")

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed text: {e}") from e

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        This method is more efficient than calling embed_text repeatedly,
        as it processes texts in batches using the model's built-in batching.

        Args:
            texts: A list of texts to embed.

        Returns:
            A 2D numpy array of shape (len(texts), 768) containing embedding vectors.

        Raises:
            ValueError: If texts is empty, not a list, or contains invalid items.
            RuntimeError: If embedding generation fails.
        """
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts)}")

        if not texts:
            raise ValueError("texts list cannot be empty")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("texts cannot contain empty or whitespace-only strings")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to embed batch: {e}") from e
