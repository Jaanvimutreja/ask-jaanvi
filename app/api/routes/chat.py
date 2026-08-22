"""
Ask Jaanvi chat API.

Pipeline:
    user query
        -> intent classification
        -> query routing
        -> knowledge retrieval
        -> trusted context construction
        -> system prompt
        -> conversation messages
        -> Groq LLM
        -> response
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.core.intent import IntentClassifier
from app.core.router import QueryRouter
from app.knowledge.loader import load_knowledge
from app.knowledge.retriever import KnowledgeRetriever
from app.llm.client import LLMClient
from app.llm.context_builder import (
    build_knowledge_context,
    build_messages,
)
from app.llm.prompt import get_system_prompt

router = APIRouter()


# Application-level components.
# Knowledge is loaded once when the module is initialized.
_knowledge = load_knowledge()
_retriever = KnowledgeRetriever(_knowledge)
_classifier = IntentClassifier()
_query_router = QueryRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a user message through the Ask Jaanvi pipeline.
    """

    # ---------------------------------------------------------
    # 1. Intent classification
    # ---------------------------------------------------------
    intent_result = _classifier.classify(request.message)

    # ---------------------------------------------------------
    # 2. Query routing
    # ---------------------------------------------------------
    decision = _query_router.route(intent_result)

    # ---------------------------------------------------------
    # 3. Knowledge retrieval
    # ---------------------------------------------------------
    knowledge_context = None

    if decision.use_knowledge:
        retrieval_result = _retriever.retrieve(
            request.message
        )

        knowledge_context = build_knowledge_context(
            [
                {
                    "section": item.section,
                    "data": item.data,
                    "matched_keywords": item.matched_keywords,
                }
                for item in retrieval_result.items
            ]
        )

    # ---------------------------------------------------------
    # 4. Select system prompt
    # ---------------------------------------------------------
    system_prompt = get_system_prompt(
        intent_result.intent.value
    )

    # ---------------------------------------------------------
    # 5. Build final LLM messages
    # ---------------------------------------------------------
    messages = build_messages(
        system_prompt=system_prompt,
        user_query=request.message,
        knowledge_context=knowledge_context,
        history=[
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.conversation_history
        ],
    )

    # ---------------------------------------------------------
    # 6. LLM call
    # ---------------------------------------------------------
    if not decision.use_llm:
        return ChatResponse(
            response=(
                "I'm Ask Jaanvi — a portfolio AI designed to discuss "
                "Jaanvi's professional background, projects, engineering "
                "approach, and technical AI/software engineering topics."
            ),
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
        )

    try:
        client = LLMClient()

        response = client.generate(
            messages,
            max_tokens=1024,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI service is currently unavailable.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="AI service request failed.",
        ) from exc

    if not response:
        raise HTTPException(
            status_code=503,
            detail="AI service returned an empty response.",
        )

    return ChatResponse(
        response=response,
        intent=intent_result.intent.value,
        confidence=intent_result.confidence,
    )