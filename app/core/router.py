"""
Query router — maps classified intents to pipeline decisions.

The router does NOT execute pipelines. It produces a RouteDecision
that tells downstream components (Phase 4/5) which pipeline to run,
whether to consult the knowledge base, and which LLM parameters to use.

Usage:
    from app.core.intent import IntentClassifier
    from app.core.router import QueryRouter

    classifier = IntentClassifier()
    router = QueryRouter()
    intent = classifier.classify("What projects have you built?")
    decision = router.route(intent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.core.intent import IntentResult, IntentType


class PipelineType(str, Enum):
    """Which response pipeline to execute."""

    IDENTITY = "identity"
    """Broad introduction — uses identity + approach from knowledge base."""

    FACTUAL = "factual"
    """Specific Jaanvi facts — uses targeted knowledge retrieval."""

    TECHNICAL = "technical"
    """General technical reasoning — LLM-driven, no knowledge injection."""

    HYBRID = "hybrid"
    """Blended — knowledge retrieval + technical reasoning."""

    META = "meta"
    """Greetings, help, off-topic — lightweight or template response."""


@dataclass
class RouteDecision:
    """
    Routing decision produced by QueryRouter.

    Consumed by the pipeline executor (Phase 5) to determine how
    to construct the prompt and call the LLM.
    """

    pipeline: PipelineType
    """Which response pipeline to run."""

    use_knowledge: bool
    """Whether to retrieve from the Jaanvi knowledge base."""

    knowledge_sections_hint: list[str] = field(default_factory=list)
    """
    Hint for which knowledge sections to prioritise.
    Empty means 'let the retriever decide based on the query'.
    """

    use_llm: bool = True
    """Whether an LLM call is needed. False for simple template responses."""

    llm_temperature: float = 0.3
    """
    LLM temperature. Lower for factual accuracy, higher for
    creative technical reasoning.
    """

    system_prompt_key: str = "default"
    """
    Key identifying which system prompt template to use.
    Resolved by the prompt builder in Phase 4.
    """


# ── Route Table ──────────────────────────────────────────────

_ROUTE_TABLE: dict[IntentType, RouteDecision] = {
    IntentType.IDENTITY_FACTUAL: RouteDecision(
        pipeline=PipelineType.IDENTITY,
        use_knowledge=True,
        knowledge_sections_hint=["identity", "approach"],
        use_llm=True,
        llm_temperature=0.3,
        system_prompt_key="identity",
    ),
    IntentType.JAANVI_FACTUAL: RouteDecision(
        pipeline=PipelineType.FACTUAL,
        use_knowledge=True,
        knowledge_sections_hint=[],  # retriever decides
        use_llm=True,
        llm_temperature=0.3,
        system_prompt_key="factual",
    ),
    IntentType.TECHNICAL_REASONING: RouteDecision(
        pipeline=PipelineType.TECHNICAL,
        use_knowledge=False,
        knowledge_sections_hint=[],
        use_llm=True,
        llm_temperature=0.7,
        system_prompt_key="technical",
    ),
    IntentType.HYBRID: RouteDecision(
        pipeline=PipelineType.HYBRID,
        use_knowledge=True,
        knowledge_sections_hint=[],  # retriever decides
        use_llm=True,
        llm_temperature=0.5,
        system_prompt_key="hybrid",
    ),
    IntentType.OFF_TOPIC: RouteDecision(
        pipeline=PipelineType.META,
        use_knowledge=False,
        knowledge_sections_hint=[],
        use_llm=False,
        llm_temperature=0.3,
        system_prompt_key="meta",
    ),
}


class QueryRouter:
    """
    Maps an IntentResult to a RouteDecision.

    Stateless — safe to reuse across requests.
    """

    def route(self, intent_result: IntentResult) -> RouteDecision:
        """
        Produce a routing decision for the given classified intent.

        Args:
            intent_result: Output from the IntentClassifier.

        Returns:
            A RouteDecision describing which pipeline, knowledge,
            LLM settings, and prompt template to use.
        """
        base = _ROUTE_TABLE.get(
            intent_result.intent,
            _ROUTE_TABLE[IntentType.OFF_TOPIC],
        )

        # Return a fresh copy so callers can modify without affecting
        # the shared route table.
        return RouteDecision(
            pipeline=base.pipeline,
            use_knowledge=base.use_knowledge,
            knowledge_sections_hint=list(base.knowledge_sections_hint),
            use_llm=base.use_llm,
            llm_temperature=base.llm_temperature,
            system_prompt_key=base.system_prompt_key,
        )
