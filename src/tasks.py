"""
Task definitions for Bug Triage environment.

Three tasks: task_criticality, task_severity, task_root_cause_assignee.
All tasks use single-step episodes.
"""

from typing import List, Dict, Optional

TASK_DEFINITIONS = {
    "task_criticality": {
        "name": "Criticality Detection",
        "difficulty": "easy",
        "description": "Determine if a bug report is critical or non-critical",
        "grader_fn": "grade_criticality",
        "required_action_fields": ["criticality"],
        "max_steps": 1,
        "reward_range": (0.0, 1.0),
    },
    "task_severity": {
        "name": "Severity Scoring",
        "difficulty": "medium",
        "description": "Assign severity score 1-5 to a bug report",
        "grader_fn": "grade_severity",
        "required_action_fields": ["severity"],
        "max_steps": 1,
        "reward_range": (0.0, 1.0),
    },
    "task_root_cause_assignee": {
        "name": "Root Cause & Assignee",
        "difficulty": "hard",
        "description": "Identify root cause category and recommend assignee",
        "grader_fn": "grade_root_cause_assignee",
        "required_action_fields": ["root_cause", "assignee"],
        "max_steps": 1,
        "reward_range": (0.0, 1.0),
    },
}


def get_task(task_id: str) -> Optional[Dict]:
    """Get task definition by ID. Returns None if not found."""
    return TASK_DEFINITIONS.get(task_id)


def get_all_task_ids() -> List[str]:
    """Return list of all task IDs."""
    return list(TASK_DEFINITIONS.keys())


def get_required_fields(task_id: str) -> List[str]:
    """Get required action fields for a task."""
    task = TASK_DEFINITIONS.get(task_id)
    return task["required_action_fields"] if task else []


def get_task_difficulty(task_id: str) -> str:
    """Get difficulty level for a task."""
    task = TASK_DEFINITIONS.get(task_id)
    return task["difficulty"] if task else "unknown"


def get_max_steps(task_id: str) -> int:
    """Get max steps for a task (all are single-step)."""
    task = TASK_DEFINITIONS.get(task_id)
    return task["max_steps"] if task else 1


def validate_task_id(task_id: str) -> bool:
    """Check if a task_id is valid."""
    return task_id in TASK_DEFINITIONS


def get_grader_fn_name(task_id: str) -> Optional[str]:
    """Get the grader function name for a task."""
    task = TASK_DEFINITIONS.get(task_id)
    return task["grader_fn"] if task else None
