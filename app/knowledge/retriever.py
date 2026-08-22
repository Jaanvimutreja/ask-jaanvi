"""
Deterministic knowledge retriever (V1).

Retrieves relevant sections from the loaded Jaanvi knowledge base
using keyword and section matching. No embeddings, no vector
databases, no ML — just structured lookup on a small knowledge base.

This module is independent of FastAPI and the LLM layer.

Usage:
    from app.knowledge.loader import load_knowledge
    from app.knowledge.retriever import KnowledgeRetriever

    knowledge = load_knowledge()
    retriever = KnowledgeRetriever(knowledge)
    result = retriever.retrieve("What projects have you built?")
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.knowledge.loader import JaanviKnowledge


# ── Result Types ─────────────────────────────────────────────────


@dataclass
class RetrievedItem:
    """A single piece of retrieved knowledge with provenance."""

    section: str
    """Top-level knowledge section (e.g. 'projects', 'skills')."""

    data: dict[str, Any]
    """The item's full data as a dict, including status field."""

    matched_keywords: list[str]
    """Which query keywords caused this item to match."""


@dataclass
class RetrievalResult:
    """Complete result of a knowledge retrieval query."""

    query: str
    """The original user query."""

    items: list[RetrievedItem] = field(default_factory=list)
    """Matched items ordered by relevance (most relevant first)."""

    sections_consulted: list[str] = field(default_factory=list)
    """Which top-level sections contributed matches."""


# ── Stopwords ────────────────────────────────────────────────────
#
# Minimal set — removes noise words without stripping topic-relevant
# terms. Intentionally small for V1 to avoid over-filtering.

_STOPWORDS = frozenset(
    {
        # Articles / determiners
        "a", "an", "the", "this", "that", "these", "those",
        # Be / have / do
        "is", "are", "was", "were", "be", "been", "being", "am",
        "have", "has", "had", "do", "does", "did",
        # Modals
        "will", "would", "could", "should", "can", "may", "might",
        # Prepositions
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "about", "through", "after", "over", "under",
        # Conjunctions
        "and", "but", "or", "not", "so", "if",
        # Pronouns
        "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their",
        # Common verbs that add noise to knowledge queries
        "tell", "know", "think", "like", "get", "got", "go",
        "make", "made", "see", "take", "give", "use", "used",
        # Filler
        "just", "also", "very", "really", "some", "any", "all",
        "more", "most", "other", "each", "every", "than", "too",
        "only", "own", "same", "much", "many", "well", "even",
        "been", "being", "then", "when", "where", "how", "what",
        "which", "who", "whom",
    }
)

# ── Section Aliases ──────────────────────────────────────────────
#
# Common words that should also match a knowledge section, beyond
# the literal section name. Keeps retrieval intuitive for users.

_SECTION_ALIASES: dict[str, list[str]] = {
    "education": ["school", "university", "college", "degree", "academic", "studied"],
    "skills": ["technologies", "tech", "stack", "tools", "languages", "frameworks"],
    "projects": ["project", "built", "created", "developed", "portfolio"],
    "experience": ["work", "job", "jobs", "career", "role", "intern", "internship"],
    "achievements": ["awards", "accomplishments", "honors", "won", "prize"],
    "interests": ["hobby", "hobbies", "goals", "passionate", "curious"],
    "identity": ["about", "profile", "introduction", "background", "yourself"],
    "approach": ["methodology", "philosophy", "values", "thinking", "mindset"],
}


# ── Retriever ────────────────────────────────────────────────────


class KnowledgeRetriever:
    """
    Deterministic retriever that matches user queries against
    the structured Jaanvi knowledge base using keyword lookup.

    Builds a keyword index at initialization and performs fast
    lookups on each query. All returned items preserve their
    original status metadata.
    """

    def __init__(self, knowledge: JaanviKnowledge) -> None:
        self._knowledge = knowledge
        self._index: dict[str, set[tuple[str, int]]] = defaultdict(set)
        self._build_index()

    # ── Public API ───────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        max_items: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve knowledge items relevant to the query.

        Args:
            query:     The user's natural-language query.
            max_items: Maximum number of items to return.

        Returns:
            A RetrievalResult with matched items sorted by relevance
            (most keyword hits first). Returns an empty result when
            no relevant information exists.
        """
        keywords = self._tokenize(query)

        if not keywords:
            return RetrievalResult(query=query)

        # Count keyword hits per (section, item_key) reference.
        hit_map: dict[tuple[str, int], set[str]] = defaultdict(set)

        for kw in keywords:
            for ref in self._index.get(kw, set()):
                hit_map[ref].add(kw)

        if not hit_map:
            return RetrievalResult(query=query)

        # Sort by number of matching keywords (descending), then
        # by section name for stable ordering.
        ranked = sorted(
            hit_map.items(),
            key=lambda pair: (-len(pair[1]), pair[0][0]),
        )

        items: list[RetrievedItem] = []
        sections: set[str] = set()

        for (section, item_key), matched_kws in ranked[:max_items]:
            data = self._get_item_data(section, item_key)
            if data is not None:
                items.append(
                    RetrievedItem(
                        section=section,
                        data=data,
                        matched_keywords=sorted(matched_kws),
                    )
                )
                sections.add(section)

        return RetrievalResult(
            query=query,
            items=items,
            sections_consulted=sorted(sections),
        )

    # ── Index Building ───────────────────────────────────────

    def _build_index(self) -> None:
        """
        Build the keyword → (section, item_key) index.

        Indexes every text field across all knowledge sections.
        Section aliases are also indexed so users can use natural
        terms like "work" instead of "experience".
        """
        k = self._knowledge

        # ── Identity (single item, key=0) ────────────────────
        self._index_item(
            "identity", 0,
            k.identity.name, k.identity.role,
            k.identity.summary, k.identity.location,
        )

        # ── Education ────────────────────────────────────────
        for i, entry in enumerate(k.education):
            self._index_item(
                "education", i,
                entry.institution, entry.degree,
                entry.field, entry.dates,
            )

        # ── Skills ───────────────────────────────────────────
        for i, cat in enumerate(k.skills):
            texts = [cat.category] + [item.name for item in cat.items]
            self._index_item("skills", i, *texts)

        # ── Projects ─────────────────────────────────────────
        for i, proj in enumerate(k.projects):
            self._index_item(
                "projects", i,
                proj.name, proj.description,
                *proj.technologies, *proj.outcomes,
            )

        # ── Experience ───────────────────────────────────────
        for i, exp in enumerate(k.experience):
            self._index_item(
                "experience", i,
                exp.role, exp.organization,
                exp.dates, *exp.highlights,
            )

        # ── Achievements ─────────────────────────────────────
        for i, ach in enumerate(k.achievements):
            self._index_item(
                "achievements", i,
                ach.title, ach.description, ach.date,
            )

        # ── Interests ────────────────────────────────────────
        for i, interest in enumerate(k.interests):
            self._index_item(
                "interests", i,
                interest.topic, interest.context,
            )

        # ── Approach (single item, key=0) ────────────────────
        self._index_item(
            "approach", 0,
            k.approach.problem_solving,
            k.approach.learning_philosophy,
            *k.approach.engineering_values,
        )

        # ── Section name aliases ─────────────────────────────
        self._index_section_aliases()

    def _index_item(
        self, section: str, item_key: int, *texts: str
    ) -> None:
        """Add all keywords from texts to the index for (section, item_key)."""
        ref = (section, item_key)
        for kw in self._extract_keywords(*texts):
            self._index[kw].add(ref)

    def _index_section_aliases(self) -> None:
        """
        Index section aliases so that natural terms like 'work'
        match all items in the 'experience' section.
        """
        section_items = {
            "identity": [0] if self._knowledge.identity else [],
            "education": list(range(len(self._knowledge.education))),
            "skills": list(range(len(self._knowledge.skills))),
            "projects": list(range(len(self._knowledge.projects))),
            "experience": list(range(len(self._knowledge.experience))),
            "achievements": list(range(len(self._knowledge.achievements))),
            "interests": list(range(len(self._knowledge.interests))),
            "approach": [0],
        }

        for section, aliases in _SECTION_ALIASES.items():
            item_keys = section_items.get(section, [])
            for alias in aliases:
                for item_key in item_keys:
                    self._index[alias].add((section, item_key))
            # Also index the section name itself as an alias.
            for item_key in item_keys:
                self._index[section].add((section, item_key))

    # ── Data Retrieval ───────────────────────────────────────

    def _get_item_data(
        self, section: str, item_key: int
    ) -> dict[str, Any] | None:
        """
        Return the dict representation of a knowledge item.

        Uses model_dump() so all fields — including status — are
        preserved in the output.
        """
        k = self._knowledge

        try:
            if section == "identity":
                return k.identity.model_dump()
            elif section == "approach":
                return k.approach.model_dump()
            elif section == "education":
                return k.education[item_key].model_dump()
            elif section == "skills":
                return k.skills[item_key].model_dump()
            elif section == "projects":
                return k.projects[item_key].model_dump()
            elif section == "experience":
                return k.experience[item_key].model_dump()
            elif section == "achievements":
                return k.achievements[item_key].model_dump()
            elif section == "interests":
                return k.interests[item_key].model_dump()
        except (IndexError, KeyError):
            return None

        return None

    # ── Tokenization ─────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text into lowercase keywords.

        Preserves characters like # and + so that terms such as
        'C++' and 'C#' survive tokenization. Removes stopwords
        and single-character tokens.

        Returns deduplicated keywords in stable order.
        """
        words = re.findall(r"[a-z0-9#+.]+", text.lower())
        seen: set[str] = set()
        result: list[str] = []
        for w in words:
            if w not in _STOPWORDS and len(w) > 1 and w not in seen:
                seen.add(w)
                result.append(w)
        return result

    @staticmethod
    def _extract_keywords(*texts: str) -> list[str]:
        """Extract and deduplicate keywords from multiple text strings."""
        keywords: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if not text:
                continue
            for kw in KnowledgeRetriever._tokenize(text):
                if kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
        return keywords
