"""
Rule-based intent classifier (V1).

Classifies user queries into one of five intents using keyword
and phrase pattern matching. Deterministic — no ML or LLM calls.

This module is independent of FastAPI and the LLM layer.
The classifier is behind a clean interface so it can be replaced
with an LLM-based classifier in a future version.

Usage:
    from app.core.intent import IntentClassifier
    classifier = IntentClassifier()
    result = classifier.classify("What projects have you built?")
    print(result.intent, result.confidence)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ── Intent Types ─────────────────────────────────────────────────


class IntentType(str, Enum):
    """The five supported query intents."""

    IDENTITY_FACTUAL = "identity_factual"
    """Broad identity / introduction questions: 'Who are you?'"""

    JAANVI_FACTUAL = "jaanvi_factual"
    """Specific factual questions about Jaanvi: 'What projects have you built?'"""

    TECHNICAL_REASONING = "technical_reasoning"
    """General technical / AI / ML questions: 'How would you design a cache?'"""

    HYBRID = "hybrid"
    """Blends Jaanvi facts with technical reasoning: 'How did you build your ML pipeline?'"""

    OFF_TOPIC = "off_topic"
    """Greetings, meta questions, or unrelated content."""


# ── Result Type ──────────────────────────────────────────────────


@dataclass
class IntentResult:
    """Typed result returned by the classifier."""

    intent: IntentType
    """The classified intent."""

    confidence: float
    """Heuristic confidence score in [0.0, 1.0]."""

    query: str
    """The original, unmodified user query."""

    matched_signals: list[str] = field(default_factory=list)
    """Which patterns or keywords contributed to the classification."""


# ═══════════════════════════════════════════════════════════════
# Signal Definitions
# ═══════════════════════════════════════════════════════════════
#
# Phrases are checked via substring match on the lowered query.
# Keywords are checked via token membership after tokenisation.

# ── Greeting / Meta (→ OFF_TOPIC) ────────────────────────────

_GREETING_PHRASES: list[str] = [
    "hello", "hi there", "hey there", "good morning",
    "good afternoon", "good evening", "howdy",
    "thank you", "thanks a lot", "thanks for",
    "goodbye", "bye bye", "see you",
]

_GREETING_TOKENS: set[str] = {
    "hi", "hey", "hello", "thanks", "goodbye", "bye",
}

_META_PHRASES: list[str] = [
    "what can you do",
    "how do you work",
    "what are you capable of",
    "what are your capabilities",
    "what should i ask",
    "how does this work",
    "can you help me",
]

# ── Identity (→ IDENTITY_FACTUAL) ────────────────────────────

_IDENTITY_PHRASES: list[str] = [
    "who are you",
    "tell me about yourself",
    "introduce yourself",
    "describe yourself",
    "what do you do",
    "what is your role",
    "what's your role",
    "what is your story",
    "what's your story",
    "give me an introduction",
    "who is jaanvi",
    "who's jaanvi",
    "what are you about",
    "what defines you",
    "your introduction",
    "brief about yourself",
]

# ── Jaanvi Factual Signals ───────────────────────────────────

_JAANVI_SECTION_KEYWORDS: set[str] = {
    "projects", "project", "portfolio",
    "skills", "skill", "technologies", "tech",
    "education", "degree", "university", "college", "school",
    "experience", "internship", "internships", "job",
    "achievements", "awards", "certifications", "honors",
    "interests", "hobbies",
    "resume", "cv", "qualifications",
}

_JAANVI_REFERENCE_TOKENS: set[str] = {
    "your", "you", "jaanvi", "jaanvi's",
}

_JAANVI_ACTION_PHRASES: list[str] = [
    "have you built",
    "have you worked",
    "have you done",
    "have you made",
    "have you created",
    "have you completed",
    "have you studied",
    "have you learned",
    "have you used",
    "did you build",
    "did you work",
    "did you study",
    "did you use",
    "do you know",
    "do you have experience",
    "are you familiar with",
    "are you experienced",
    "what have you",
    "where did you",
    "where do you",
    "when did you",
    "your experience with",
    "your work on",
    "your background in",
]

# ── Technical Signals ────────────────────────────────────────

_TECHNICAL_PHRASES: list[str] = [
    "how would you approach",
    "how would you design",
    "how would you build",
    "how would you implement",
    "how would you solve",
    "how would you handle",
    "how would you optimize",
    "how to implement",
    "how to build",
    "how to design",
    "how to deploy",
    "how to scale",
    "how does",
    "how do",
    "explain how",
    "explain what",
    "explain the concept",
    "explain the difference",
    "what is the best way to",
    "what is the difference between",
    "what's the difference between",
    "what are the trade-offs",
    "what are the pros and cons",
    "compare and contrast",
    "design a system",
    "design an api",
    "design a database",
    "design a pipeline",
    "walk me through",
    "step by step",
    "best practices for",
]

_TECHNICAL_KEYWORDS: set[str] = {
    "algorithm", "algorithms",
    "architecture", "microservice", "microservices",
    "api", "rest", "graphql", "grpc",
    "database", "sql", "nosql", "postgresql", "mongodb",
    "deploy", "deployment", "kubernetes", "docker", "ci/cd",
    "optimize", "optimization", "scalability", "latency",
    "cache", "caching", "redis",
    "implement", "implementation",
    "neural", "transformer", "embeddings", "llm", "gpt",
    "gradient", "backpropagation", "regularization",
    "cnn", "rnn", "lstm", "gan", "vae", "bert",
    "training", "inference", "fine-tuning", "finetuning",
    "supervised", "unsupervised", "reinforcement",
    "regression", "classification", "clustering",
    "precision", "recall", "f1",
    "pipeline", "etl", "workflow",
    "distributed", "concurrent", "parallel",
    "encryption", "authentication", "authorization",
    "complexity", "big-o",
    "testing", "tdd", "unit-test",
    "agile", "scrum",
    "data-structure", "tree", "graph", "heap",
    "sorting", "searching", "hashing",
}

# ── Hybrid Detection Phrases ─────────────────────────────────
#
# These phrases combine Jaanvi reference with technical inquiry.

_HYBRID_PHRASES: list[str] = [
    "how did you build",
    "how did you approach",
    "how did you design",
    "how did you implement",
    "how did you solve",
    "what approach did you",
    "what approach do you",
    "what technology did you",
    "what technologies did you",
    "why did you choose",
    "why did you use",
    "walk me through your",
    "explain your approach",
    "explain your project",
    "tell me about your approach to",
    "how does your project",
    "how does your system",
    "improve your",
    "improve upon your",
]


# ═══════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════


class IntentClassifier:
    """
    Rule-based intent classifier.

    Normalises the query, matches against keyword and phrase signals,
    scores each intent category, and returns the dominant intent with
    a heuristic confidence score.

    This class is stateless and safe to reuse across requests.
    """

    # ── Public API ───────────────────────────────────────────

    def classify(self, query: str) -> IntentResult:
        """
        Classify a user query into an intent.

        Args:
            query: The raw user query string.

        Returns:
            An IntentResult with intent, confidence, original query,
            and which signals matched.
        """
        if not query or not query.strip():
            return IntentResult(
                intent=IntentType.OFF_TOPIC,
                confidence=0.9,
                query=query,
                matched_signals=["empty_query"],
            )

        query_lower = query.lower().strip()
        tokens = self._tokenize(query_lower)
        signals: dict[str, list[str]] = {
            "greeting": [],
            "meta": [],
            "identity": [],
            "jaanvi": [],
            "technical": [],
            "hybrid": [],
        }

        # ── 1. Collect signals ───────────────────────────────

        self._match_greeting_signals(query_lower, tokens, signals)
        self._match_identity_signals(query_lower, signals)
        self._match_jaanvi_signals(query_lower, tokens, signals)
        self._match_technical_signals(query_lower, tokens, signals)
        self._match_hybrid_signals(query_lower, signals)

        # ── 2. Score each category ───────────────────────────

        scores = self._compute_scores(signals, tokens)

        # ── 3. Decide intent ─────────────────────────────────

        return self._decide(scores, signals, query)

    # ── Signal Matchers ──────────────────────────────────────

    @staticmethod
    def _match_greeting_signals(
        query_lower: str,
        tokens: list[str],
        signals: dict[str, list[str]],
    ) -> None:
        for phrase in _GREETING_PHRASES:
            if phrase in query_lower:
                signals["greeting"].append(f"phrase:{phrase}")
        for token in tokens:
            if token in _GREETING_TOKENS:
                signals["greeting"].append(f"token:{token}")
        for phrase in _META_PHRASES:
            if phrase in query_lower:
                signals["meta"].append(f"phrase:{phrase}")

    @staticmethod
    def _match_identity_signals(
        query_lower: str,
        signals: dict[str, list[str]],
    ) -> None:
        for phrase in _IDENTITY_PHRASES:
            if phrase in query_lower:
                signals["identity"].append(f"phrase:{phrase}")

    @staticmethod
    def _match_jaanvi_signals(
        query_lower: str,
        tokens: list[str],
        signals: dict[str, list[str]],
    ) -> None:
        # Possessive / reference tokens
        refs_found = [t for t in tokens if t in _JAANVI_REFERENCE_TOKENS]
        for ref in refs_found:
            signals["jaanvi"].append(f"ref:{ref}")

        # Section keywords
        sections_found = [t for t in tokens if t in _JAANVI_SECTION_KEYWORDS]
        for sec in sections_found:
            signals["jaanvi"].append(f"section:{sec}")

        # Action phrases ("have you built", etc.)
        for phrase in _JAANVI_ACTION_PHRASES:
            if phrase in query_lower:
                signals["jaanvi"].append(f"action:{phrase}")

    @staticmethod
    def _match_technical_signals(
        query_lower: str,
        tokens: list[str],
        signals: dict[str, list[str]],
    ) -> None:
        for phrase in _TECHNICAL_PHRASES:
            if phrase in query_lower:
                signals["technical"].append(f"phrase:{phrase}")
        for token in tokens:
            if token in _TECHNICAL_KEYWORDS:
                signals["technical"].append(f"keyword:{token}")

    @staticmethod
    def _match_hybrid_signals(
        query_lower: str,
        signals: dict[str, list[str]],
    ) -> None:
        for phrase in _HYBRID_PHRASES:
            if phrase in query_lower:
                signals["hybrid"].append(f"phrase:{phrase}")

    # ── Scoring ──────────────────────────────────────────────

    @staticmethod
    def _compute_scores(
        signals: dict[str, list[str]],
        tokens: list[str],
    ) -> dict[str, float]:
        """
        Convert raw signal counts into weighted scores.

        Phrases are weighted higher than individual keywords because
        they carry stronger intent signal.
        """
        def _weighted_count(signal_list: list[str]) -> float:
            score = 0.0
            for s in signal_list:
                if s.startswith("phrase:") or s.startswith("action:"):
                    score += 2.0
                elif s.startswith("section:"):
                    score += 1.5
                else:  # token:, ref:, keyword:
                    score += 1.0
            return score

        scores: dict[str, float] = {}
        scores["greeting"] = _weighted_count(signals["greeting"]) + _weighted_count(signals["meta"])
        scores["identity"] = _weighted_count(signals["identity"])
        scores["jaanvi"] = _weighted_count(signals["jaanvi"])
        scores["technical"] = _weighted_count(signals["technical"])
        scores["hybrid"] = _weighted_count(signals["hybrid"])

        # Boost hybrid if both jaanvi and technical have signals
        if signals["jaanvi"] and signals["technical"]:
            scores["hybrid"] += 1.5

        return scores

    # ── Decision Logic ───────────────────────────────────────

    @staticmethod
    def _decide(
        scores: dict[str, float],
        signals: dict[str, list[str]],
        original_query: str,
    ) -> IntentResult:
        """Apply decision rules to produce the final IntentResult."""

        all_matched = []
        for category_signals in signals.values():
            all_matched.extend(category_signals)

        # ── Greeting / meta detection ────────────────────────
        # Greeting/meta wins unless there are strong Jaanvi signals
        # (section keywords or action phrases — not just bare 'you').
        has_strong_jaanvi = any(
            s.startswith("section:") or s.startswith("action:")
            for s in signals["jaanvi"]
        )
        if scores["greeting"] > 0 and not has_strong_jaanvi and scores["technical"] == 0 and scores["identity"] == 0:
            return IntentResult(
                intent=IntentType.OFF_TOPIC,
                confidence=0.9,
                query=original_query,
                matched_signals=signals["greeting"] + signals["meta"],
            )

        # ── Explicit hybrid phrases take priority
        if scores["hybrid"] >= 2.0:
            conf = min(0.6 + scores["hybrid"] * 0.05, 0.9)
            return IntentResult(
                intent=IntentType.HYBRID,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=all_matched,
            )

        # ── Both jaanvi and technical signals present → hybrid
        if scores["jaanvi"] >= 2.0 and scores["technical"] >= 2.0:
            total = scores["jaanvi"] + scores["technical"]
            conf = min(0.55 + total * 0.03, 0.85)
            return IntentResult(
                intent=IntentType.HYBRID,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=all_matched,
            )

        # ── Identity dominates
        if scores["identity"] >= 2.0 and scores["identity"] >= scores["jaanvi"]:
            conf = min(0.6 + scores["identity"] * 0.08, 0.95)
            return IntentResult(
                intent=IntentType.IDENTITY_FACTUAL,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=signals["identity"],
            )

        # ── Jaanvi factual dominates
        if scores["jaanvi"] >= 2.0 and scores["jaanvi"] > scores["technical"]:
            conf = min(0.6 + scores["jaanvi"] * 0.05, 0.9)
            return IntentResult(
                intent=IntentType.JAANVI_FACTUAL,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=signals["jaanvi"],
            )

        # ── Technical dominates
        if scores["technical"] >= 1.0:
            conf = min(0.5 + scores["technical"] * 0.06, 0.9)
            return IntentResult(
                intent=IntentType.TECHNICAL_REASONING,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=signals["technical"],
            )

        # ── Weak jaanvi signal (e.g. just a section keyword)
        if scores["jaanvi"] >= 1.0:
            conf = min(0.4 + scores["jaanvi"] * 0.05, 0.7)
            return IntentResult(
                intent=IntentType.JAANVI_FACTUAL,
                confidence=round(conf, 2),
                query=original_query,
                matched_signals=signals["jaanvi"],
            )

        # ── Weak identity signal
        if scores["identity"] > 0:
            return IntentResult(
                intent=IntentType.IDENTITY_FACTUAL,
                confidence=0.5,
                query=original_query,
                matched_signals=signals["identity"],
            )

        # ── Fallback: off-topic
        return IntentResult(
            intent=IntentType.OFF_TOPIC,
            confidence=0.4,
            query=original_query,
            matched_signals=["no_matching_signals"],
        )

    # ── Tokenisation ─────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Split lowered text into tokens.

        Preserves characters like -, /, #, + so terms such as
        'ci/cd', 'c++', 'c#', 'big-o' survive tokenisation.
        """
        return re.findall(r"[a-z0-9#+/\-]+", text)
