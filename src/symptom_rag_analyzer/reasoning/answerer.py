"""Application interface for sending clinical context and evidence to the reasoning LLM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SYSTEM_PROMPT, build_prompt


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "openai/gpt-5.6-luna"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_OUTPUT_TOKENS = 4096


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================

_RAG_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnostic_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string"
                    },
                    "rank": {
                        "type": "integer"
                    },
                    "justification": {
                        "type": "string"
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                    },
                },
                "required": [
                    "condition",
                    "rank",
                    "justification",
                    "supporting_evidence",
                ],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "string"
        },
    },
    "required": [
        "diagnostic_candidates",
        "limitations",
    ],
    "additionalProperties": False,
}


# =============================================================================
# INTERNAL DATA MODELS
# =============================================================================

@dataclass
class DiagnosticCandidate:
    """One preliminary diagnostic hypothesis supported by retrieved evidence."""

    condition: str
    rank: int
    justification: str
    supporting_evidence: list[int]


@dataclass
class RAGAnswer:
    """Internal Answerer output, not yet the final common Data Fusion output."""

    diagnostic_candidates: list[DiagnosticCandidate]
    limitations: str


# =============================================================================
# BIOMEDICAL ANSWERER
# =============================================================================

class BiomedicalAnswerer:
    """Send prepared clinical context to OpenRouter and validate its response."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Initialize the OpenRouter client using OPENROUTER_API_KEY."""

        load_dotenv()

        api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key or not api_key.strip():
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required."
            )

        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "model must be a non-empty string."
            )

        self.model = model

        try:
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize the OpenRouter client."
            ) from error

    # -------------------------------------------------------------------------
    # ANSWER
    # -------------------------------------------------------------------------

    def answer(
        self,
        clinical_context: str,
        evidence_context: str,
    ) -> RAGAnswer:
        """Generate and validate an internal RAG answer."""

        if not isinstance(clinical_context, str):
            raise TypeError(
                "clinical_context must be a string."
            )

        if not isinstance(evidence_context, str):
            raise TypeError(
                "evidence_context must be a string."
            )

        if not clinical_context.strip():
            raise ValueError(
                "clinical_context cannot be empty or whitespace."
            )

        prompt = build_prompt(
            clinical_context,
            evidence_context,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                # The Answerer only needs a small structured response.
                max_tokens=MAX_OUTPUT_TOKENS,

                # Deterministic reasoning output.
                temperature=0,

                # Request structured JSON matching our RAG contract.
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rag_answer",
                        "strict": True,
                        "schema": _RAG_ANSWER_SCHEMA,
                    },
                },

                # Allow OpenRouter to select a provider endpoint that can
                # satisfy the request. This avoids the routing failure we
                # encountered with require_parameters=True.
                extra_body={
                    "provider": {
                        "require_parameters": False,
                    }
                },
            )

        except Exception as error:
            raise RuntimeError(
                "OpenRouter request failed."
            ) from error

        # ---------------------------------------------------------------------
        # Extract model response
        # ---------------------------------------------------------------------

        try:
            response_text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "Invalid OpenRouter response: missing message content."
            ) from error

        if not response_text:
            raise ValueError(
                "Invalid OpenRouter response: response content was empty."
            )

        return self._parse_response(response_text)

    # -------------------------------------------------------------------------
    # RESPONSE PARSER
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_response(response_text: str) -> RAGAnswer:
        """Validate OpenRouter JSON and convert it to RAGAnswer."""

        try:
            payload: Any = json.loads(response_text)

        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Invalid OpenRouter response: response was not valid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise ValueError(
                "Invalid OpenRouter response: expected a JSON object."
            )

        candidates = payload.get(
            "diagnostic_candidates"
        )

        limitations = payload.get(
            "limitations"
        )

        if not isinstance(candidates, list):
            raise ValueError(
                "Invalid OpenRouter response: "
                "diagnostic_candidates must be a list."
            )

        if not isinstance(limitations, str):
            raise ValueError(
                "Invalid OpenRouter response: "
                "limitations must be a string."
            )

        # ---------------------------------------------------------------------
        # Validate every diagnostic candidate
        # ---------------------------------------------------------------------

        validated_candidates: list[DiagnosticCandidate] = []

        for candidate in candidates:

            if not isinstance(candidate, dict):
                raise ValueError(
                    "Invalid OpenRouter response: "
                    "each candidate must be an object."
                )

            condition = candidate.get(
                "condition"
            )

            rank = candidate.get(
                "rank"
            )

            justification = candidate.get(
                "justification"
            )

            supporting_evidence = candidate.get(
                "supporting_evidence"
            )

            if (
                not isinstance(condition, str)
                or not condition.strip()
            ):
                raise ValueError(
                    "Invalid OpenRouter response: "
                    "condition must be non-empty."
                )

            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
            ):
                raise ValueError(
                    "Invalid OpenRouter response: "
                    "rank must be an integer."
                )

            if (
                not isinstance(justification, str)
                or not justification.strip()
            ):
                raise ValueError(
                    "Invalid OpenRouter response: "
                    "justification must be non-empty."
                )

            if (
                not isinstance(supporting_evidence, list)
                or any(
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    for item in supporting_evidence
                )
            ):
                raise ValueError(
                    "Invalid OpenRouter response: "
                    "supporting_evidence must contain integers."
                )

            validated_candidates.append(
                DiagnosticCandidate(
                    condition=condition,
                    rank=rank,
                    justification=justification,
                    supporting_evidence=supporting_evidence,
                )
            )

        # ---------------------------------------------------------------------
        # Return existing internal contract
        # ---------------------------------------------------------------------

        return RAGAnswer(
            diagnostic_candidates=validated_candidates,
            limitations=limitations,
        )