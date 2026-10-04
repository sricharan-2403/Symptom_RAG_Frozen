import json

from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter
from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_full_symptom_rag_agent_with_myocardial_infarction_case():
    # ---------------------------------------------------------
    # 1. CTC-style structured input
    # ---------------------------------------------------------
    ctc_output = {
        "diseases": ["Myocardial Infarction"],
        "symptoms": [
            "chest pain",
            "sweating",
            "shortness of breath",
        ],
        "medications": ["Sildenafil"],
        "tests": [],
    }

    clinical_context = ClinicalTextClarifierOutput(
        clinical_text=(
            "The patient has myocardial infarction with chest pain, "
            "sweating, and shortness of breath. "
            "The patient is taking Sildenafil."
        ),
        diseases=ctc_output["diseases"],
        symptoms=ctc_output["symptoms"],
        medications=ctc_output["medications"],
        tests=ctc_output["tests"],
    )

    # ---------------------------------------------------------
    # 2. Create a small biomedical knowledge base
    # ---------------------------------------------------------
    document = Document(
        filename="cardiology_guidelines.pdf",
        page_number=25,
        text=(
            "Myocardial infarction commonly presents with chest pain, "
            "shortness of breath, and sweating."
        ),
        source_type="PDF",
    )

    chunk = Chunk(
        document=document,
        chunk_index=0,
        text=document.text,
    )

    # ---------------------------------------------------------
    # 3. Create biomedical embedding + in-memory Qdrant
    # ---------------------------------------------------------
    embedding_model = BiomedicalEmbeddingModel()

    chunk_embedding = embedding_model.embed_text(chunk.text)

    vector_store = QdrantVectorStore(
        collection_name="manual_full_agent_test",
        vector_size=768,
        client=QdrantClient(":memory:"),
    )

    vector_store.insert(
        [chunk],
        [chunk_embedding],
    )

    # ---------------------------------------------------------
    # 4. Build the Symptom RAG agent
    # ---------------------------------------------------------
    retriever = BiomedicalRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    orchestrator = SymptomRAGOrchestrator(
        retriever=retriever,
        context_builder=EvidenceContextBuilder(),
        answerer=BiomedicalAnswerer(),
        top_k=1,
    )

    # ---------------------------------------------------------
    # 5. Verify the retrieval query
    # ---------------------------------------------------------
    query = orchestrator._build_retrieval_query(clinical_context)

    assert "Diseases: Myocardial Infarction" in query
    assert (
        "Symptoms: chest pain, sweating, shortness of breath"
        in query
    )

    # Current design intentionally excludes medications/tests
    # from the retrieval query.
    assert "Sildenafil" not in query
    assert "medications" not in query.lower()
    assert "tests" not in query.lower()

    # ---------------------------------------------------------
    # 6. Run the complete RAG orchestration
    # ---------------------------------------------------------
    result = orchestrator.analyze(clinical_context)

    assert result.answer is not None
    assert result.evidence

    # ---------------------------------------------------------
    # 7. Convert internal result to common agent contract
    # ---------------------------------------------------------
    output = SymptomRAGOutputAdapter().adapt(
        clinical_context,
        result,
    )

    # ---------------------------------------------------------
    # 8. Validate final agent output
    # ---------------------------------------------------------
    assert output["agent"] == "symptom_rag"
    assert output["status"] == "success"

    assert output["query_context"] == {
        "diseases": ["Myocardial Infarction"],
        "symptoms": [
            "chest pain",
            "sweating",
            "shortness of breath",
        ],
        "medications": ["Sildenafil"],
        "tests": [],
        "procedures": [],
    }

    assert isinstance(output["diagnostic_candidates"], list)
    assert output["diagnostic_candidates"]

    candidate = output["diagnostic_candidates"][0]

    assert isinstance(candidate["condition"], str)
    assert candidate["condition"].strip()

    assert isinstance(candidate["rank"], int)

    assert isinstance(candidate["justification"], str)
    assert candidate["justification"].strip()

    assert isinstance(candidate["supporting_evidence"], list)

    # ---------------------------------------------------------
    # 9. Validate retrieved evidence
    # ---------------------------------------------------------
    assert isinstance(output["evidence"], list)
    assert output["evidence"]

    evidence = output["evidence"][0]

    assert evidence["source"] == "cardiology_guidelines.pdf"
    assert evidence["source_type"] == "PDF"
    assert evidence["location"] == "Page 25"

    assert (
        "Myocardial infarction" in evidence["content"]
    )

    assert isinstance(
        evidence["relevance_score"],
        (int, float),
    )

    # ---------------------------------------------------------
    # 10. Validate metadata and error state
    # ---------------------------------------------------------
    assert output["metadata"]["evidence_count"] == len(
        output["evidence"]
    )

    assert isinstance(output["limitations"], str)

    assert output["error"] is None

    # ---------------------------------------------------------
    # 11. Verify the final result is JSON serializable
    # ---------------------------------------------------------
    serialized = json.dumps(
        output,
        indent=2,
    )

    print("\n========== FINAL SYMPTOM RAG AGENT OUTPUT ==========")
    print(serialized)
    print("====================================================")