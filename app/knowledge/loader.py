"""
Knowledge base loader and schema definitions.

Loads Jaanvi's structured knowledge from YAML into validated Python
objects. All knowledge models are defined here so the schema is
co-located with the loading logic.

This module is independent of FastAPI and the LLM layer.

Usage:
    from app.knowledge.loader import load_knowledge
    knowledge = load_knowledge()            # default path
    knowledge = load_knowledge("custom.yaml")  # custom path
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# Default path to the knowledge base, relative to this module.
DEFAULT_KNOWLEDGE_PATH = Path(__file__).parent / "jaanvi.yaml"


# ── Exceptions ───────────────────────────────────────────────────


class KnowledgeLoadError(Exception):
    """Raised when the knowledge base cannot be loaded or validated."""


# ── Status Enum ──────────────────────────────────────────────────


class FactStatus(str, Enum):
    """
    Truth status for every factual claim about Jaanvi.

    The conversational AI must respect these statuses:
      - demonstrated, current, past → may be stated as accomplished fact.
      - learning, interest, planned → must be explicitly qualified.
      - failed                      → may be mentioned with context.
      - unknown                     → must not be presented as verified.
    """

    DEMONSTRATED = "demonstrated"
    CURRENT = "current"
    PAST = "past"
    LEARNING = "learning"
    INTEREST = "interest"
    PLANNED = "planned"
    FAILED = "failed"
    UNKNOWN = "unknown"


# ── Knowledge Models ─────────────────────────────────────────────
#
# All models use extra="forbid" so that misspelled or unexpected
# YAML keys cause a clear validation error instead of being silently
# ignored.


class _StrictModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class Metadata(_StrictModel):
    """Knowledge base metadata."""

    version: str
    last_updated: str
    truth_policy: str


class Identity(_StrictModel):
    """Jaanvi's core identity information."""

    name: str
    role: str = ""
    summary: str = ""
    location: str = ""


class EducationEntry(_StrictModel):
    """A single education record."""

    institution: str
    degree: str
    field: str = ""
    dates: str = ""
    status: FactStatus


class SkillItem(_StrictModel):
    """An individual skill within a category."""

    name: str
    status: FactStatus


class SkillCategory(_StrictModel):
    """A group of related skills."""

    category: str
    items: list[SkillItem] = Field(default_factory=list)


class ProjectEntry(_StrictModel):
    """A project Jaanvi has worked on or is planning."""

    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    status: FactStatus


class ExperienceEntry(_StrictModel):
    """A professional or work experience record."""

    role: str
    organization: str
    dates: str = ""
    highlights: list[str] = Field(default_factory=list)
    status: FactStatus


class AchievementEntry(_StrictModel):
    """An achievement, award, or accomplishment."""

    title: str
    description: str = ""
    date: str = ""
    status: FactStatus


class InterestEntry(_StrictModel):
    """A topic Jaanvi is interested in or exploring."""

    topic: str
    context: str = ""
    status: FactStatus


class Approach(_StrictModel):
    """Jaanvi's engineering approach and values."""

    problem_solving: str = ""
    learning_philosophy: str = ""
    engineering_values: list[str] = Field(default_factory=list)


class JaanviKnowledge(_StrictModel):
    """
    Root model for Jaanvi's complete knowledge base.

    All factual claims about Jaanvi flow through this structure.
    The loader validates incoming YAML against this schema.
    """

    metadata: Metadata
    identity: Identity
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[SkillCategory] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    achievements: list[AchievementEntry] = Field(default_factory=list)
    interests: list[InterestEntry] = Field(default_factory=list)
    approach: Approach = Field(default_factory=Approach)

    @field_validator(
        "education",
        "skills",
        "projects",
        "experience",
        "achievements",
        "interests",
        mode="before",
    )
    @classmethod
    def coerce_none_to_empty_list(cls, v: Any) -> list:
        """Accept YAML null / bare key as an empty list."""
        return [] if v is None else v

    @field_validator("approach", mode="before")
    @classmethod
    def coerce_none_to_empty_approach(cls, v: Any) -> dict:
        """Accept YAML null for approach as empty defaults."""
        return {} if v is None else v


# ── Loader ───────────────────────────────────────────────────────


def load_knowledge(
    path: Path | str = DEFAULT_KNOWLEDGE_PATH,
) -> JaanviKnowledge:
    """
    Load and validate the Jaanvi knowledge base from a YAML file.

    Args:
        path: Path to the YAML knowledge file.

    Returns:
        A validated JaanviKnowledge instance.

    Raises:
        KnowledgeLoadError: If the file is missing, malformed, or
            fails schema validation.
    """
    path = Path(path)

    # 1. File existence
    if not path.exists():
        raise KnowledgeLoadError(f"Knowledge file not found: {path}")
    if not path.is_file():
        raise KnowledgeLoadError(f"Knowledge path is not a file: {path}")

    # 2. YAML parsing
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise KnowledgeLoadError(
            f"Invalid YAML syntax in {path.name}: {exc}"
        ) from exc

    # 3. Top-level structure
    if raw is None:
        raise KnowledgeLoadError(f"Knowledge file {path.name} is empty.")
    if not isinstance(raw, dict):
        raise KnowledgeLoadError(
            f"Knowledge file {path.name} must contain a YAML mapping "
            f"at the top level, got {type(raw).__name__}."
        )

    # 4. Schema validation
    try:
        knowledge = JaanviKnowledge.model_validate(raw)
    except Exception as exc:
        raise KnowledgeLoadError(
            f"Knowledge schema validation failed for {path.name}:\n{exc}"
        ) from exc

    logger.info(
        "Knowledge base loaded: %d education, %d skill categories, "
        "%d projects, %d experience, %d achievements, %d interests",
        len(knowledge.education),
        len(knowledge.skills),
        len(knowledge.projects),
        len(knowledge.experience),
        len(knowledge.achievements),
        len(knowledge.interests),
    )

    return knowledge
