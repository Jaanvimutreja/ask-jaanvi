"""
Tests for intent classification and query routing.

Each test class targets a specific intent category with representative
queries, plus edge cases for ambiguous and degenerate inputs.
"""

import pytest

from app.core.intent import IntentClassifier, IntentResult, IntentType
from app.core.router import PipelineType, QueryRouter, RouteDecision


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def classifier() -> IntentClassifier:
    return IntentClassifier()


@pytest.fixture()
def router() -> QueryRouter:
    return QueryRouter()


# ═══════════════════════════════════════════════════════════════
# Intent Result Structure
# ═══════════════════════════════════════════════════════════════


class TestIntentResultStructure:
    """Every classification must return a well-formed IntentResult."""

    def test_returns_intent_result(self, classifier):
        result = classifier.classify("anything")
        assert isinstance(result, IntentResult)

    def test_intent_is_valid_enum(self, classifier):
        result = classifier.classify("hello")
        assert isinstance(result.intent, IntentType)

    def test_confidence_in_range(self, classifier):
        for query in ["hello", "your projects", "design a system", "", "xyz"]:
            result = classifier.classify(query)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of range for: {query!r}"
            )

    def test_original_query_preserved(self, classifier):
        raw = "  What Projects Have You Built?  "
        result = classifier.classify(raw)
        assert result.query == raw

    def test_matched_signals_populated(self, classifier):
        result = classifier.classify("What are your projects?")
        assert len(result.matched_signals) > 0


# ═══════════════════════════════════════════════════════════════
# Identity Factual
# ═══════════════════════════════════════════════════════════════


class TestIdentityFactual:
    """Broad identity / introduction questions → IDENTITY_FACTUAL."""

    @pytest.mark.parametrize(
        "query",
        [
            "Who are you?",
            "Tell me about yourself",
            "Introduce yourself",
            "Describe yourself",
            "What do you do?",
            "What is your role?",
            "Who is Jaanvi?",
            "What is your story?",
            "Give me an introduction",
        ],
    )
    def test_identity_queries(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.IDENTITY_FACTUAL, (
            f"Expected IDENTITY_FACTUAL for: {query!r}, got {result.intent}"
        )

    def test_identity_confidence_is_reasonable(self, classifier):
        result = classifier.classify("Who are you?")
        assert result.confidence >= 0.5


# ═══════════════════════════════════════════════════════════════
# Jaanvi Factual
# ═══════════════════════════════════════════════════════════════


class TestJaanviFactual:
    """Specific factual questions about Jaanvi → JAANVI_FACTUAL."""

    @pytest.mark.parametrize(
        "query",
        [
            "What projects have you built?",
            "Tell me about your projects",
            "What are your skills?",
            "What technologies do you know?",
            "Where did you study?",
            "What is your education?",
            "Do you have experience with Python?",
            "What internships have you done?",
            "Tell me about your work experience",
            "What achievements do you have?",
            "Have you worked with machine learning?",
            "What are your qualifications?",
            "Have you built any AI projects?",
            "Are you familiar with FastAPI?",
            "What is your tech stack?",
        ],
    )
    def test_jaanvi_factual_queries(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.JAANVI_FACTUAL, (
            f"Expected JAANVI_FACTUAL for: {query!r}, got {result.intent}"
        )

    def test_jaanvi_factual_confidence_is_reasonable(self, classifier):
        result = classifier.classify("What projects have you built?")
        assert result.confidence >= 0.5

    def test_possessive_plus_section_keyword(self, classifier):
        """'your' + section keyword should trigger jaanvi_factual."""
        result = classifier.classify("your education")
        assert result.intent in (IntentType.JAANVI_FACTUAL, IntentType.IDENTITY_FACTUAL)

    def test_jaanvi_name_reference(self, classifier):
        """Direct reference to 'Jaanvi' should trigger a Jaanvi intent."""
        result = classifier.classify("What has Jaanvi studied?")
        assert result.intent in (
            IntentType.JAANVI_FACTUAL,
            IntentType.IDENTITY_FACTUAL,
        )


# ═══════════════════════════════════════════════════════════════
# Technical Reasoning
# ═══════════════════════════════════════════════════════════════


class TestTechnicalReasoning:
    """General technical / AI / ML questions → TECHNICAL_REASONING."""

    @pytest.mark.parametrize(
        "query",
        [
            "How would you design a distributed cache?",
            "Explain gradient descent",
            "How to implement a REST API?",
            "What is the difference between SQL and NoSQL?",
            "How would you approach building a recommendation system?",
            "What are the trade-offs of microservices?",
            "Design a system for real-time notifications",
            "How does a transformer architecture work?",
            "What is the best way to deploy a machine learning model?",
            "Compare supervised and unsupervised learning",
            "How would you optimize database queries?",
            "Explain backpropagation step by step",
            "Best practices for API design",
        ],
    )
    def test_technical_queries(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.TECHNICAL_REASONING, (
            f"Expected TECHNICAL_REASONING for: {query!r}, got {result.intent}"
        )

    def test_technical_confidence_is_reasonable(self, classifier):
        result = classifier.classify("How would you design a distributed cache?")
        assert result.confidence >= 0.5

    def test_pure_technical_no_jaanvi_reference(self, classifier):
        """Technical question without 'your/you' should not be jaanvi."""
        result = classifier.classify("What is a neural network?")
        assert result.intent == IntentType.TECHNICAL_REASONING


# ═══════════════════════════════════════════════════════════════
# Hybrid
# ═══════════════════════════════════════════════════════════════


class TestHybrid:
    """Blended Jaanvi + technical questions → HYBRID."""

    @pytest.mark.parametrize(
        "query",
        [
            "How did you build your ML Pipeline project?",
            "What approach did you use for your NLP project?",
            "Explain your approach to API design",
            "How did you implement authentication in your project?",
            "Why did you choose FastAPI for your project?",
            "Walk me through your system design",
            "How does your project handle scaling?",
        ],
    )
    def test_hybrid_queries(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.HYBRID, (
            f"Expected HYBRID for: {query!r}, got {result.intent}"
        )

    def test_hybrid_confidence_is_reasonable(self, classifier):
        result = classifier.classify("How did you build your ML Pipeline?")
        assert result.confidence >= 0.4


# ═══════════════════════════════════════════════════════════════
# Off-Topic
# ═══════════════════════════════════════════════════════════════


class TestOffTopic:
    """Greetings, meta, and unrelated questions → OFF_TOPIC."""

    @pytest.mark.parametrize(
        "query",
        [
            "Hello!",
            "Hi there",
            "Hey",
            "Good morning",
            "Thanks!",
            "Thank you so much",
            "Goodbye",
            "What can you do?",
        ],
    )
    def test_greeting_and_meta(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.OFF_TOPIC, (
            f"Expected OFF_TOPIC for: {query!r}, got {result.intent}"
        )

    @pytest.mark.parametrize(
        "query",
        [
            "What is the weather today?",
            "Tell me a joke",
            "Who won the world cup?",
            "random gibberish xyz abc",
        ],
    )
    def test_unrelated_queries(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == IntentType.OFF_TOPIC, (
            f"Expected OFF_TOPIC for: {query!r}, got {result.intent}"
        )

    def test_off_topic_confidence(self, classifier):
        result = classifier.classify("Hello!")
        assert result.confidence >= 0.5


# ═══════════════════════════════════════════════════════════════
# Edge Cases & Ambiguity
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Empty, very short, and ambiguous inputs."""

    def test_empty_string(self, classifier):
        result = classifier.classify("")
        assert result.intent == IntentType.OFF_TOPIC
        assert result.confidence >= 0.5

    def test_whitespace_only(self, classifier):
        result = classifier.classify("   ")
        assert result.intent == IntentType.OFF_TOPIC

    def test_single_word_section_keyword(self, classifier):
        """A bare section keyword like 'projects' should lean factual."""
        result = classifier.classify("projects")
        assert result.intent in (
            IntentType.JAANVI_FACTUAL,
            IntentType.OFF_TOPIC,
        )

    def test_single_technical_keyword(self, classifier):
        """A single technical keyword should lean technical."""
        result = classifier.classify("algorithm")
        assert result.intent in (
            IntentType.TECHNICAL_REASONING,
            IntentType.OFF_TOPIC,
        )

    def test_case_insensitivity(self, classifier):
        """Classification should be case-insensitive."""
        lower = classifier.classify("who are you?")
        upper = classifier.classify("WHO ARE YOU?")
        mixed = classifier.classify("Who Are You?")
        assert lower.intent == upper.intent == mixed.intent

    def test_deterministic(self, classifier):
        """Same input should always produce the same output."""
        query = "What projects have you built?"
        results = [classifier.classify(query) for _ in range(5)]
        intents = {r.intent for r in results}
        confs = {r.confidence for r in results}
        assert len(intents) == 1, "Non-deterministic intent"
        assert len(confs) == 1, "Non-deterministic confidence"

    def test_classifier_is_stateless(self, classifier):
        """Multiple calls should not affect each other."""
        r1 = classifier.classify("Hello!")
        r2 = classifier.classify("What are your skills?")
        r3 = classifier.classify("Hello!")
        assert r1.intent == r3.intent
        assert r1.confidence == r3.confidence

    def test_very_long_query(self, classifier):
        """A very long query should not crash."""
        long_query = "What are your skills? " * 100
        result = classifier.classify(long_query)
        assert isinstance(result, IntentResult)


# ═══════════════════════════════════════════════════════════════
# Router Tests
# ═══════════════════════════════════════════════════════════════


class TestQueryRouter:
    """Tests for the intent → pipeline routing logic."""

    def test_identity_routes_to_identity_pipeline(self, classifier, router):
        intent = classifier.classify("Who are you?")
        decision = router.route(intent)
        assert decision.pipeline == PipelineType.IDENTITY

    def test_jaanvi_factual_routes_to_factual_pipeline(self, classifier, router):
        intent = classifier.classify("What projects have you built?")
        decision = router.route(intent)
        assert decision.pipeline == PipelineType.FACTUAL

    def test_technical_routes_to_technical_pipeline(self, classifier, router):
        intent = classifier.classify("How would you design a distributed cache?")
        decision = router.route(intent)
        assert decision.pipeline == PipelineType.TECHNICAL

    def test_hybrid_routes_to_hybrid_pipeline(self, classifier, router):
        intent = classifier.classify("How did you build your ML Pipeline project?")
        decision = router.route(intent)
        assert decision.pipeline == PipelineType.HYBRID

    def test_off_topic_routes_to_meta_pipeline(self, classifier, router):
        intent = classifier.classify("Hello!")
        decision = router.route(intent)
        assert decision.pipeline == PipelineType.META

    def test_returns_route_decision(self, router, classifier):
        intent = classifier.classify("anything")
        decision = router.route(intent)
        assert isinstance(decision, RouteDecision)

    def test_factual_uses_knowledge(self, classifier, router):
        intent = classifier.classify("What are your skills?")
        decision = router.route(intent)
        assert decision.use_knowledge is True

    def test_technical_does_not_use_knowledge(self, classifier, router):
        intent = classifier.classify("How does gradient descent work?")
        decision = router.route(intent)
        assert decision.use_knowledge is False

    def test_factual_temperature_is_low(self, classifier, router):
        intent = classifier.classify("Tell me about your projects")
        decision = router.route(intent)
        assert decision.llm_temperature <= 0.4

    def test_technical_temperature_is_higher(self, classifier, router):
        intent = classifier.classify("Design a system for real-time analytics")
        decision = router.route(intent)
        assert decision.llm_temperature >= 0.5

    def test_meta_does_not_use_llm(self, classifier, router):
        intent = classifier.classify("Hello!")
        decision = router.route(intent)
        assert decision.use_llm is False

    def test_identity_has_section_hints(self, classifier, router):
        intent = classifier.classify("Who are you?")
        decision = router.route(intent)
        assert "identity" in decision.knowledge_sections_hint

    def test_route_returns_independent_copies(self, router, classifier):
        """Two route calls should return independent objects."""
        intent = classifier.classify("Who are you?")
        d1 = router.route(intent)
        d2 = router.route(intent)
        d1.knowledge_sections_hint.append("mutated")
        assert "mutated" not in d2.knowledge_sections_hint

    def test_every_intent_type_has_a_route(self, router):
        """Every IntentType must produce a valid RouteDecision."""
        for intent_type in IntentType:
            fake_result = IntentResult(
                intent=intent_type,
                confidence=0.8,
                query="test",
            )
            decision = router.route(fake_result)
            assert isinstance(decision, RouteDecision)
            assert isinstance(decision.pipeline, PipelineType)
