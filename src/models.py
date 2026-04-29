"""
Pydantic models for Bug Report Triage RL Environment.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from enum import Enum
from openenv.core.env_server.types import Action, Observation


class CriticalityLabel(str, Enum):
    """Task 1: Criticality Detection"""
    CRITICAL = "critical"
    NON_CRITICAL = "non_critical"


class SeverityLevel(int, Enum):
    """Task 2: Severity Scoring (5-point scale)"""
    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class RootCauseCategory(str, Enum):
    """Task 3: Root Cause Classification"""
    BUG = "bug"
    DESIGN = "design"
    ENVIRONMENT = "environment"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    EXTERNAL = "external"


class BugReport(BaseModel):
    """A GitHub bug report."""
    bug_id: str
    title: str
    body: str
    labels: List[str] = []
    created_at: str
    repo: str
    comments_text: List[str] = []
    author: str
    is_pull_request: bool = False

    model_config = ConfigDict(frozen=False)


class BugTriageObservation(Observation):
    """What the agent sees at each step."""
    task_id: str  # "task_criticality", "task_severity", "task_root_cause_assignee"
    bug_report: BugReport
    available_assignees: List[str] = []  # Populated for task 3
    step: int
    max_steps: int
    done: bool = False

    model_config = ConfigDict(frozen=False)


class BugTriageAction(Action):
    """What the agent returns as its decision."""
    task_id: str
    bug_id: str

    # Task 1: Criticality
    criticality: Optional[CriticalityLabel] = None

    # Task 2: Severity
    severity: Optional[SeverityLevel] = None

    # Task 3: Root Cause + Assignee
    root_cause: Optional[RootCauseCategory] = None
    assignee: Optional[str] = None

    # Metadata
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""

    model_config = ConfigDict(frozen=False)


class BugTriageReward(BaseModel):
    """Reward signal returned to the agent."""
    base_score: float = Field(ge=0.0, le=1.0)
    confidence_bonus: float = 0.0
    reasoning_bonus: float = 0.0
    edge_case_bonus: float = 0.0
    total: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=False)


class BugGroundTruth(BaseModel):
    """Ground truth labels (internal to environment, hidden from agent)."""
    bug_id: str
    criticality: CriticalityLabel
    severity: SeverityLevel
    root_cause: RootCauseCategory
    assignee: str
    is_ambiguous: bool = False

    model_config = ConfigDict(frozen=False)


__all__ = [
    "CriticalityLabel",
    "SeverityLevel",
    "RootCauseCategory",
    "BugReport",
    "BugTriageObservation",
    "BugTriageAction",
    "BugTriageReward",
    "BugGroundTruth",
]
