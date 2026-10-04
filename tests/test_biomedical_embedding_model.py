"""Simple test script for BiomedicalEmbeddingModel."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel


def main():
    """Test BiomedicalEmbeddingModel."""
    print("Initializing BiomedicalEmbeddingModel...")
    model = BiomedicalEmbeddingModel()
    
    # Embed a single text
    text = "Pneumonia commonly causes fever and cough."
    print(f"\nEmbedding text: '{text}'")
    embedding = model.embed_text(text)
    
    # Print results
    print(f"\nEmbedding vector:\n{embedding}")
    print(f"\nVector type: {type(embedding)}")
    print(f"Vector shape: {embedding.shape}")
    print(f"Embedding dimension: {model.embedding_dim}")


if __name__ == "__main__":
    main()
