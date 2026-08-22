"""
Tests for the knowledge base loader and retriever.

Uses synthetic test data — NOT Jaanvi's real information.
All fixtures write temporary YAML files via pytest's tmp_path.
"""

import pytest
import yaml

from app.knowledge.loader import (
    FactStatus,
    JaanviKnowledge,
    KnowledgeLoadError,
    load_knowledge,
)
from app.knowledge.retriever import KnowledgeRetriever, RetrievalResult


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

# Minimal valid knowledge structure with synthetic test data.
VALID_KNOWLEDGE_DATA = {
    "metadata": {
        "version": "1.0",
        "last_updated": "2026-01-01",
        "truth_policy": "Test policy — only verified facts.",
    },
    "identity": {
        "name": "TestUser",
        "role": "AI Engineer",
        "summary": "A test engineer for unit tests.",
        "location": "Test City",
    },
    "education": [
        {
            "institution": "Test University",
            "degree": "B.Tech",
            "field": "Computer Science",
            "dates": "2020 - 2024",
            "status": "demonstrated",
        },
    ],
    "skills": [
        {
            "category": "Programming Languages",
            "items": [
                {"name": "Python", "status": "demonstrated"},
                {"name": "Rust", "status": "learning"},
            ],
        },
        {
            "category": "Frameworks",
            "items": [
                {"name": "FastAPI", "status": "demonstrated"},
                {"name": "PyTorch", "status": "interest"},
            ],
        },
    ],
    "projects": [
        {
            "name": "ML Pipeline",
            "description": "An end-to-end machine learning pipeline for NLP tasks",
            "technologies": ["Python", "scikit-learn", "FastAPI"],
            "outcomes": ["Reduced inference latency"],
            "status": "demonstrated",
        },
        {
            "name": "Robotics Navigator",
            "description": "Autonomous robot navigation system",
            "technologies": ["ROS", "Python", "C++"],
            "outcomes": [],
            "status": "planned",
        },
    ],
    "experience": [
        {
            "role": "AI Research Intern",
            "organization": "Tech Corp",
            "dates": "Summer 2024",
            "highlights": ["Developed NLP classification models"],
            "status": "past",
        },
    ],
    "achievements": [
        {
            "title": "Hackathon Winner",
            "description": "First place in AI hackathon",
            "date": "2024",
            "status": "demonstrated",
        },
        {
            "title": "Startup Attempt",
            "description": "Attempted to build an ed-tech startup",
            "date": "2023",
            "status": "failed",
        },
    ],
    "interests": [
        {
            "topic": "Reinforcement Learning",
            "context": "Applying RL to robotics control",
            "status": "interest",
        },
    ],
    "approach": {
        "problem_solving": "Break complex problems into smaller testable components",
        "learning_philosophy": "Learn by building real projects",
        "engineering_values": ["Clean code", "Testing", "Documentation"],
    },
}


def _write_yaml(tmp_path, data, filename="knowledge.yaml"):
    """Write a dict as YAML to a temp file and return the path."""
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path


@pytest.fixture()
def valid_yaml_path(tmp_path):
    """Write the valid test knowledge to a temp YAML file."""
    return _write_yaml(tmp_path, VALID_KNOWLEDGE_DATA)


@pytest.fixture()
def knowledge(valid_yaml_path) -> JaanviKnowledge:
    """Load the valid test knowledge."""
    return load_knowledge(valid_yaml_path)


@pytest.fixture()
def retriever(knowledge) -> KnowledgeRetriever:
    """Build a retriever from the valid test knowledge."""
    return KnowledgeRetriever(knowledge)


# ═══════════════════════════════════════════════════════════════
# Loader Tests
# ═══════════════════════════════════════════════════════════════


class TestKnowledgeLoaderValid:
    """Tests for successful loading of well-formed YAML."""

    def test_valid_yaml_loads_successfully(self, knowledge):
        """A structurally valid YAML file should load without errors."""
        assert isinstance(knowledge, JaanviKnowledge)

    def test_metadata_loaded(self, knowledge):
        """Metadata section should be fully populated."""
        assert knowledge.metadata.version == "1.0"
        assert knowledge.metadata.last_updated == "2026-01-01"
        assert "verified" in knowledge.metadata.truth_policy.lower()

    def test_identity_loaded(self, knowledge):
        """Identity section should preserve all fields."""
        assert knowledge.identity.name == "TestUser"
        assert knowledge.identity.role == "AI Engineer"

    def test_list_sections_loaded(self, knowledge):
        """All list-based sections should load with correct counts."""
        assert len(knowledge.education) == 1
        assert len(knowledge.skills) == 2
        assert len(knowledge.projects) == 2
        assert len(knowledge.experience) == 1
        assert len(knowledge.achievements) == 2
        assert len(knowledge.interests) == 1

    def test_nested_skill_items_loaded(self, knowledge):
        """Skill categories should contain their nested items."""
        lang_category = knowledge.skills[0]
        assert lang_category.category == "Programming Languages"
        assert len(lang_category.items) == 2
        assert lang_category.items[0].name == "Python"

    def test_approach_loaded(self, knowledge):
        """Approach section should preserve all fields."""
        assert "testable" in knowledge.approach.problem_solving.lower()
        assert len(knowledge.approach.engineering_values) == 3


class TestKnowledgeLoaderStatuses:
    """Tests that status fields are correctly parsed and preserved."""

    def test_demonstrated_status(self, knowledge):
        assert knowledge.education[0].status == FactStatus.DEMONSTRATED

    def test_past_status(self, knowledge):
        assert knowledge.experience[0].status == FactStatus.PAST

    def test_learning_status(self, knowledge):
        rust_skill = knowledge.skills[0].items[1]
        assert rust_skill.name == "Rust"
        assert rust_skill.status == FactStatus.LEARNING

    def test_interest_status(self, knowledge):
        assert knowledge.interests[0].status == FactStatus.INTEREST

    def test_planned_status(self, knowledge):
        assert knowledge.projects[1].status == FactStatus.PLANNED

    def test_failed_status(self, knowledge):
        assert knowledge.achievements[1].status == FactStatus.FAILED

    def test_all_status_values_are_valid_enum_members(self, knowledge):
        """Every status in the loaded data should be a FactStatus member."""
        statuses = []
        for entry in knowledge.education:
            statuses.append(entry.status)
        for cat in knowledge.skills:
            for item in cat.items:
                statuses.append(item.status)
        for proj in knowledge.projects:
            statuses.append(proj.status)
        for exp in knowledge.experience:
            statuses.append(exp.status)
        for ach in knowledge.achievements:
            statuses.append(ach.status)
        for interest in knowledge.interests:
            statuses.append(interest.status)

        for status in statuses:
            assert isinstance(status, FactStatus), (
                f"Expected FactStatus, got {type(status)}: {status}"
            )


class TestKnowledgeLoaderInvalid:
    """Tests that malformed or invalid YAML fails clearly."""

    def test_missing_file_raises_error(self, tmp_path):
        """A non-existent file should raise KnowledgeLoadError."""
        with pytest.raises(KnowledgeLoadError, match="not found"):
            load_knowledge(tmp_path / "does_not_exist.yaml")

    def test_empty_file_raises_error(self, tmp_path):
        """An empty YAML file should raise KnowledgeLoadError."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(KnowledgeLoadError, match="empty"):
            load_knowledge(path)

    def test_non_mapping_raises_error(self, tmp_path):
        """A YAML file with a list at top level should fail."""
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(KnowledgeLoadError, match="mapping"):
            load_knowledge(path)

    def test_malformed_yaml_raises_error(self, tmp_path):
        """Invalid YAML syntax should raise KnowledgeLoadError."""
        path = tmp_path / "bad.yaml"
        path.write_text("metadata:\n  version: [unterminated\n")
        with pytest.raises(KnowledgeLoadError, match="YAML"):
            load_knowledge(path)

    def test_missing_required_section_raises_error(self, tmp_path):
        """Missing required 'metadata' section should fail validation."""
        data = {
            "identity": {"name": "Test"},
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(KnowledgeLoadError, match="validation failed"):
            load_knowledge(path)

    def test_missing_required_field_raises_error(self, tmp_path):
        """Missing required field within a section should fail."""
        data = {
            "metadata": {
                "version": "1.0",
                "last_updated": "2026-01-01",
                "truth_policy": "Test",
            },
            "identity": {
                # 'name' is required but missing
                "role": "Engineer",
            },
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(KnowledgeLoadError, match="validation failed"):
            load_knowledge(path)

    def test_invalid_status_value_raises_error(self, tmp_path):
        """An unrecognized status value should fail validation."""
        data = dict(VALID_KNOWLEDGE_DATA)
        data = {**VALID_KNOWLEDGE_DATA}
        data["projects"] = [
            {
                "name": "Bad Project",
                "status": "invented_status",
            }
        ]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(KnowledgeLoadError, match="validation failed"):
            load_knowledge(path)

    def test_extra_field_rejected(self, tmp_path):
        """Unexpected fields should be rejected (strict mode)."""
        data = {**VALID_KNOWLEDGE_DATA}
        data["identity"] = {
            "name": "Test",
            "role": "Engineer",
            "invented_field": "should fail",
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(KnowledgeLoadError, match="validation failed"):
            load_knowledge(path)

    def test_null_list_sections_coerced_to_empty(self, tmp_path):
        """Sections set to null/None in YAML should become empty lists."""
        data = {
            "metadata": {
                "version": "1.0",
                "last_updated": "2026-01-01",
                "truth_policy": "Test",
            },
            "identity": {"name": "Test"},
            "education": None,
            "skills": None,
            "projects": None,
            "experience": None,
            "achievements": None,
            "interests": None,
            "approach": None,
        }
        path = _write_yaml(tmp_path, data)
        knowledge = load_knowledge(path)
        assert knowledge.education == []
        assert knowledge.skills == []
        assert knowledge.projects == []
        assert knowledge.experience == []
        assert knowledge.achievements == []
        assert knowledge.interests == []


class TestKnowledgeLoaderProductionYAML:
    """Tests that the actual jaanvi.yaml file loads correctly."""

    def test_production_yaml_loads(self):
        """The shipped jaanvi.yaml should load without errors."""
        from app.knowledge.loader import DEFAULT_KNOWLEDGE_PATH

        knowledge = load_knowledge(DEFAULT_KNOWLEDGE_PATH)
        assert isinstance(knowledge, JaanviKnowledge)
        assert knowledge.identity.name == "Jaanvi Mutreja"


# ═══════════════════════════════════════════════════════════════
# Retriever Tests
# ═══════════════════════════════════════════════════════════════


class TestRetrieverRelevantQueries:
    """Tests that relevant queries retrieve correct sections."""

    def test_project_keyword_retrieves_projects(self, retriever):
        """A query mentioning a project name should return that project."""
        result = retriever.retrieve("Tell me about the ML Pipeline")
        sections = result.sections_consulted
        assert "projects" in sections
        project_items = [i for i in result.items if i.section == "projects"]
        assert any("ML Pipeline" in str(i.data.get("name", "")) for i in project_items)

    def test_technology_keyword_retrieves_matching_items(self, retriever):
        """A query about a specific technology should match items using it."""
        result = retriever.retrieve("Do you work with Python?")
        assert len(result.items) > 0
        # Python appears in skills and projects
        sections = result.sections_consulted
        assert "skills" in sections or "projects" in sections

    def test_section_name_retrieves_that_section(self, retriever):
        """Querying a section name like 'education' should return its items."""
        result = retriever.retrieve("education")
        assert "education" in result.sections_consulted
        assert len(result.items) > 0

    def test_section_alias_retrieves_section(self, retriever):
        """Section aliases like 'work' should match 'experience'."""
        result = retriever.retrieve("work experience")
        assert "experience" in result.sections_consulted

    def test_skills_alias_retrieves_skills(self, retriever):
        """Alias 'technologies' should retrieve the skills section."""
        result = retriever.retrieve("What technologies do you know?")
        assert "skills" in result.sections_consulted

    def test_identity_query(self, retriever):
        """A query about the person should return identity info."""
        result = retriever.retrieve("Tell me about yourself")
        assert "identity" in result.sections_consulted

    def test_interest_query(self, retriever):
        """A query about interests should match the interests section."""
        result = retriever.retrieve("reinforcement learning")
        assert "interests" in result.sections_consulted

    def test_achievement_query(self, retriever):
        """A query about achievements should return that section."""
        result = retriever.retrieve("hackathon")
        assert "achievements" in result.sections_consulted


class TestRetrieverIrrelevantQueries:
    """Tests that unrelated queries return empty or minimal results."""

    def test_completely_unrelated_query(self, retriever):
        """A query with no matching keywords should return empty."""
        result = retriever.retrieve("quantum entanglement superconductors")
        assert len(result.items) == 0
        assert len(result.sections_consulted) == 0

    def test_stopwords_only_query(self, retriever):
        """A query of only stopwords should return empty."""
        result = retriever.retrieve("the is a an to of in for on with")
        assert len(result.items) == 0

    def test_empty_query(self, retriever):
        """An empty query should return empty."""
        result = retriever.retrieve("")
        assert len(result.items) == 0


class TestRetrieverStatusPreservation:
    """Tests that retrieved context never strips status metadata."""

    def test_demonstrated_status_preserved_in_retrieval(self, retriever):
        """Retrieved items must include the original 'status' field."""
        result = retriever.retrieve("ML Pipeline")
        project_items = [i for i in result.items if i.section == "projects"]
        assert len(project_items) > 0
        for item in project_items:
            assert "status" in item.data, (
                "Status field missing from retrieved project data"
            )

    def test_planned_status_preserved(self, retriever):
        """A 'planned' project should retain its planned status."""
        result = retriever.retrieve("Robotics Navigator")
        robot_items = [
            i for i in result.items
            if i.section == "projects"
            and i.data.get("name") == "Robotics Navigator"
        ]
        assert len(robot_items) == 1
        assert robot_items[0].data["status"] == FactStatus.PLANNED

    def test_failed_status_preserved(self, retriever):
        """A 'failed' achievement should retain its failed status."""
        result = retriever.retrieve("startup")
        startup_items = [
            i for i in result.items
            if i.section == "achievements"
            and "Startup" in str(i.data.get("title", ""))
        ]
        assert len(startup_items) == 1
        assert startup_items[0].data["status"] == FactStatus.FAILED

    def test_learning_status_preserved(self, retriever):
        """A 'learning' skill should retain its learning status."""
        result = retriever.retrieve("Rust")
        skill_items = [i for i in result.items if i.section == "skills"]
        # The skill category containing Rust should be returned
        rust_found = False
        for item in skill_items:
            for skill in item.data.get("items", []):
                if skill.get("name") == "Rust":
                    assert skill["status"] == FactStatus.LEARNING
                    rust_found = True
        assert rust_found, "Rust skill not found in retrieval results"

    def test_interest_status_preserved(self, retriever):
        """An 'interest' entry should retain its interest status."""
        result = retriever.retrieve("reinforcement learning")
        interest_items = [i for i in result.items if i.section == "interests"]
        assert len(interest_items) > 0
        assert interest_items[0].data["status"] == FactStatus.INTEREST

    def test_all_retrieved_list_items_have_status(self, retriever):
        """Every retrieved item from a status-bearing section must have status."""
        STATUS_SECTIONS = {
            "education", "projects", "experience",
            "achievements", "interests",
        }
        result = retriever.retrieve("Python education projects work hackathon")
        for item in result.items:
            if item.section in STATUS_SECTIONS:
                assert "status" in item.data, (
                    f"Status missing from {item.section} item: {item.data}"
                )


class TestRetrieverResultStructure:
    """Tests for the structure and properties of retrieval results."""

    def test_result_type(self, retriever):
        """retrieve() should return a RetrievalResult."""
        result = retriever.retrieve("anything")
        assert isinstance(result, RetrievalResult)

    def test_result_preserves_original_query(self, retriever):
        """The result should record the original query string."""
        query = "What projects have you built?"
        result = retriever.retrieve(query)
        assert result.query == query

    def test_matched_keywords_populated(self, retriever):
        """Each retrieved item should list which keywords caused the match."""
        result = retriever.retrieve("Python FastAPI")
        for item in result.items:
            assert len(item.matched_keywords) > 0

    def test_max_items_respected(self, retriever):
        """The max_items parameter should cap the number of results."""
        result = retriever.retrieve("Python", max_items=2)
        assert len(result.items) <= 2

    def test_results_ordered_by_relevance(self, retriever):
        """Items with more keyword matches should appear first."""
        # "ML Pipeline Python" should rank the ML Pipeline project
        # higher than items matching only one keyword.
        result = retriever.retrieve("ML Pipeline Python scikit-learn")
        if len(result.items) >= 2:
            first_kw_count = len(result.items[0].matched_keywords)
            second_kw_count = len(result.items[1].matched_keywords)
            assert first_kw_count >= second_kw_count


class TestRetrieverEmptyKnowledgeBase:
    """Tests retriever behavior with an empty knowledge base."""

    @pytest.fixture()
    def empty_retriever(self, tmp_path):
        """Build a retriever from a minimal/empty knowledge base."""
        data = {
            "metadata": {
                "version": "1.0",
                "last_updated": "2026-01-01",
                "truth_policy": "Test",
            },
            "identity": {"name": "Empty"},
        }
        path = _write_yaml(tmp_path, data)
        knowledge = load_knowledge(path)
        return KnowledgeRetriever(knowledge)

    def test_section_query_on_empty_returns_no_list_items(self, empty_retriever):
        """Querying 'projects' on an empty KB returns no project items."""
        result = empty_retriever.retrieve("projects")
        project_items = [i for i in result.items if i.section == "projects"]
        assert len(project_items) == 0

    def test_keyword_query_on_empty_returns_minimal(self, empty_retriever):
        """A keyword query on an empty KB should return at most identity."""
        result = empty_retriever.retrieve("Python machine learning")
        non_identity = [i for i in result.items if i.section != "identity"]
        assert len(non_identity) == 0
