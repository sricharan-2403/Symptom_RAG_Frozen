#!/usr/bin/env python
"""Manual end-to-end test for the real persistent Symptom RAG knowledge base.

This test verifies the complete RAG pipeline using:
- Real persistent Qdrant storage at data/qdrant/
- Production collection: symptom_chunks
- Full inference pipeline: retrieval → reasoning → output

Run from project root:
    python tests/test_e2e_real_rag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src/ to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.orchestrator import ClinicalTextClarifierOutput, SymptomRAGOrchestrator
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter
from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_e2e_real_rag():
    """Test the complete RAG pipeline on the real persistent knowledge base."""
    
    print("=" * 80)
    print("END-TO-END TEST: REAL PERSISTENT SYMPTOM RAG KNOWLEDGE BASE")
    print("=" * 80)
    
    # Test input
    test_input = {
        "diseases": ["Myocardial Infarction"],
        "symptoms": ["chest pain", "sweating", "shortness of breath"],
        "medications": ["Sildenafil"],
        "tests": [],
    }
    
    print("\n1. Test Input (Clinical Context)")
    print("-" * 80)
    print(json.dumps(test_input, indent=2))
    
    # Construct ClinicalTextClarifierOutput
    print("\n2. Constructing ClinicalTextClarifierOutput")
    print("-" * 80)
    clinical_context = ClinicalTextClarifierOutput(
        clinical_text=(
            f"Patient with {', '.join(test_input['diseases'])} "
            f"presenting with {', '.join(test_input['symptoms'])}. "
            f"On medication: {', '.join(test_input['medications'])}"
        ),
        diseases=test_input["diseases"],
        symptoms=test_input["symptoms"],
        medications=test_input["medications"],
        tests=test_input["tests"],
    )
    print(f"✓ ClinicalTextClarifierOutput created")
    print(f"  - clinical_text: {clinical_context.clinical_text[:80]}...")
    print(f"  - diseases: {clinical_context.diseases}")
    print(f"  - symptoms: {clinical_context.symptoms}")
    
    # Set up persistent vector store
    print("\n3. Connecting to Persistent Qdrant Storage")
    print("-" * 80)
    qdrant_path = Path("data/qdrant")
    collection_name = "symptom_chunks"
    
    if not qdrant_path.exists():
        raise FileNotFoundError(f"Persistent Qdrant storage not found: {qdrant_path}")
    
    print(f"✓ Qdrant path exists: {qdrant_path.resolve()}")
    print(f"✓ Collection name: {collection_name}")
    
    # Initialize components (using persistent Qdrant)
    print("\n4. Initializing RAG Pipeline Components")
    print("-" * 80)
    
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=768,
        path=qdrant_path,
    )
    print(f"✓ Vector Store initialized (persistent)")
    
    embedding_model = BiomedicalEmbeddingModel()
    print(f"✓ Embedding Model loaded")
    
    retriever = BiomedicalRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    print(f"✓ Retriever initialized")
    
    context_builder = EvidenceContextBuilder()
    print(f"✓ Context Builder initialized")
    
    answerer = BiomedicalAnswerer()
    print(f"✓ Answerer initialized")
    
    orchestrator = SymptomRAGOrchestrator(
        retriever=retriever,
        context_builder=context_builder,
        answerer=answerer,
        top_k=5,
    )
    print(f"✓ Orchestrator initialized")
    
    # Run the RAG pipeline
    print("\n5. Running RAG Pipeline")
    print("-" * 80)
    
    rag_result = orchestrator.analyze(clinical_context)
    
    print(f"✓ Pipeline completed")
    print(f"  - Retrieved evidence items: {len(rag_result.evidence)}")
    print(f"  - Answer status: {rag_result.answer.status if hasattr(rag_result.answer, 'status') else 'N/A'}")
    
    # Verify retrieval
    print("\n6. Verifying Retrieval")
    print("-" * 80)
    
    if not rag_result.evidence:
        raise ValueError("No evidence retrieved from the knowledge base")
    
    print(f"✓ Retrieved {len(rag_result.evidence)} evidence items")
    
    # Sample evidence
    for i, evidence in enumerate(rag_result.evidence[:3], 1):
        print(f"\n  Evidence {i}:")
        print(f"    - Score: {evidence.score:.4f}")
        print(f"    - Text: {evidence.text[:100]}...")
        print(f"    - Filename: {evidence.filename}")
        print(f"    - Chunk Index: {evidence.chunk_index}")
        print(f"    - Source Type: {evidence.source_type}")
        print(f"    - Disease: {evidence.metadata.get('disease_name', 'N/A')}")
    
    # Prepare output
    print("\n7. Preparing Output")
    print("-" * 80)
    
    output_adapter = SymptomRAGOutputAdapter()
    output_dict = output_adapter.adapt(clinical_context, rag_result)
    
    print(f"✓ Output adapter applied")
    
    # Verify output is JSON serializable
    print("\n8. Verifying Final Output")
    print("-" * 80)
    
    try:
        output_json = json.dumps(output_dict)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Output is not JSON serializable: {e}")
    
    print(f"✓ Output is valid JSON")
    
    # Verify required fields
    required_checks = {
        "agent": ("symptom_rag", output_dict.get("agent")),
        "status": ("success", output_dict.get("status")),
        "query_context": ("present", "query_context" in output_dict),
        "diagnostic_candidates": ("list", isinstance(output_dict.get("diagnostic_candidates"), list)),
        "evidence": ("non-empty list", isinstance(output_dict.get("evidence"), list) and len(output_dict.get("evidence", [])) > 0),
        "error": (None, output_dict.get("error")),
        "limitations": ("present", isinstance(output_dict.get("limitations"), str)),
        "metadata": ("present", "metadata" in output_dict),
    }
    
    print("\nField Verification:")
    for field, (expected, actual) in required_checks.items():
        if field == "query_context":
            status = "✓" if actual else "✗"
            print(f"  {status} {field}: {expected}")
        elif field == "diagnostic_candidates":
            status = "✓" if actual else "✗"
            print(f"  {status} {field}: {expected} (length: {len(output_dict.get('diagnostic_candidates', []))})")
        elif field == "evidence":
            status = "✓" if actual else "✗"
            print(f"  {status} {field}: {expected} (length: {len(output_dict.get('evidence', []))})")
        elif field == "metadata":
            status = "✓" if actual else "✗"
            print(f"  {status} {field}: {expected}")
            if actual:
                evidence_count = output_dict.get("metadata", {}).get("evidence_count")
                actual_evidence_len = len(output_dict.get("evidence", []))
                count_match = "✓" if evidence_count == actual_evidence_len else "✗"
                print(f"    {count_match} evidence_count: {evidence_count} == actual: {actual_evidence_len}")
        else:
            status = "✓" if actual == expected else "✗"
            print(f"  {status} {field}: expected={expected}, actual={actual}")
    
    # Print complete output
    print("\n9. Complete Final Output (JSON)")
    print("-" * 80)
    print(json.dumps(output_dict, indent=2))
    
    print("\n" + "=" * 80)
    print("END-TO-END TEST: COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_e2e_real_rag()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
