"""
Tests for tasks.py helper functions + env.py hardening edge cases.

Owner: Sumit
Day 5: Coverage boost for tasks.py and env.py error paths.
"""

import pytest
from src.tasks import (
    TASK_DEFINITIONS,
    get_task,
    get_all_task_ids,
    get_required_fields,
    get_task_difficulty,
    get_max_steps,
    validate_task_id,
    get_grader_fn_name,
)
from src.utils import (
    truncate_text,
    validate_bug_entry,
    validate_dataset,
    get_data_distribution,
    get_bug_by_id,
    filter_bugs_by_difficulty,
    load_json,
)


# ── tasks.py ─────────────────────────────────────────────────────────────────

def test_get_task_valid():
    t = get_task("task_criticality")
    assert t is not None
    assert t["name"] == "Criticality Detection"

def test_get_task_invalid():
    assert get_task("nonexistent") is None

def test_get_all_task_ids():
    ids = get_all_task_ids()
    assert len(ids) == 3
    assert "task_criticality" in ids
    assert "task_severity" in ids
    assert "task_root_cause_assignee" in ids

def test_get_required_fields_criticality():
    fields = get_required_fields("task_criticality")
    assert fields == ["criticality"]

def test_get_required_fields_severity():
    assert get_required_fields("task_severity") == ["severity"]

def test_get_required_fields_root_cause():
    assert get_required_fields("task_root_cause_assignee") == ["root_cause", "assignee"]

def test_get_required_fields_invalid():
    assert get_required_fields("nonexistent") == []

def test_get_task_difficulty():
    assert get_task_difficulty("task_criticality") == "easy"
    assert get_task_difficulty("task_severity") == "medium"
    assert get_task_difficulty("task_root_cause_assignee") == "hard"

def test_get_task_difficulty_invalid():
    assert get_task_difficulty("nonexistent") == "unknown"

def test_get_max_steps():
    for tid in get_all_task_ids():
        assert get_max_steps(tid) == 1

def test_get_max_steps_invalid():
    assert get_max_steps("nonexistent") == 1

def test_validate_task_id():
    assert validate_task_id("task_criticality") is True
    assert validate_task_id("task_severity") is True
    assert validate_task_id("task_root_cause_assignee") is True
    assert validate_task_id("bad") is False

def test_get_grader_fn_name():
    assert get_grader_fn_name("task_criticality") == "grade_criticality"
    assert get_grader_fn_name("task_severity") == "grade_severity"
    assert get_grader_fn_name("task_root_cause_assignee") == "grade_root_cause_assignee"

def test_get_grader_fn_name_invalid():
    assert get_grader_fn_name("nonexistent") is None

def test_task_definitions_have_all_keys():
    for tid, tdef in TASK_DEFINITIONS.items():
        assert "name" in tdef
        assert "difficulty" in tdef
        assert "description" in tdef
        assert "grader_fn" in tdef
        assert "required_action_fields" in tdef
        assert "max_steps" in tdef
        assert "reward_range" in tdef


# ── utils.py ─────────────────────────────────────────────────────────────────

def test_truncate_text_short():
    assert truncate_text("hello", 100) == "hello"

def test_truncate_text_long():
    result = truncate_text("a" * 200, 100)
    assert len(result) == 103  # 100 + "..."
    assert result.endswith("...")

def test_truncate_text_empty():
    assert truncate_text("", 100) == ""

def test_truncate_text_none():
    assert truncate_text(None, 100) == ""

def test_validate_bug_entry_valid():
    bug = {
        "bug_id": "test/repo#1", "title": "t", "body": "b",
        "labels": [], "created_at": "2024-01-01", "repo": "test/repo",
        "author": "user1",
        "ground_truth": {
            "criticality": "critical", "severity": 4,
            "root_cause": "bug", "assignee": "dev1"
        }
    }
    errors = validate_bug_entry(bug)
    assert errors == []

def test_validate_bug_entry_missing_fields():
    errors = validate_bug_entry({})
    assert len(errors) > 0

def test_validate_bug_entry_invalid_gt():
    bug = {
        "bug_id": "x", "title": "t", "body": "b",
        "labels": [], "created_at": "d", "repo": "r", "author": "a",
        "ground_truth": {"criticality": "bad", "severity": 99, "root_cause": "bad", "assignee": "a"}
    }
    errors = validate_bug_entry(bug)
    assert any("criticality" in e for e in errors)
    assert any("severity" in e for e in errors)
    assert any("root_cause" in e for e in errors)

def test_validate_dataset():
    bugs = [
        {"bug_id": "x", "title": "t", "body": "b", "labels": [],
         "created_at": "d", "repo": "r", "author": "a",
         "ground_truth": {"criticality": "critical", "severity": 3,
                          "root_cause": "bug", "assignee": "dev"}},
    ]
    result = validate_dataset(bugs)
    assert result["total"] == 1
    assert result["valid"] == 1
    assert result["invalid"] == 0

def test_get_data_distribution():
    bugs = [
        {"repo": "a/b", "ground_truth": {"criticality": "critical", "severity": 3,
                                          "root_cause": "bug", "is_ambiguous": True}},
        {"repo": "a/b", "ground_truth": {"criticality": "non_critical", "severity": 1,
                                          "root_cause": "design", "is_ambiguous": False}},
    ]
    stats = get_data_distribution(bugs)
    assert stats["total"] == 2
    assert stats["ambiguous"] == 1
    assert stats["repos"]["a/b"] == 2

def test_get_bug_by_id():
    bugs = [{"bug_id": "a"}, {"bug_id": "b"}]
    assert get_bug_by_id(bugs, "a")["bug_id"] == "a"
    assert get_bug_by_id(bugs, "c") is None

def test_load_json():
    data = load_json("data/bugs_processed.json")
    assert isinstance(data, list)
    assert len(data) >= 180

def test_filter_bugs_all():
    bugs = [{"ground_truth": {"is_ambiguous": False, "severity": 3, "root_cause": "bug"}}]
    assert len(filter_bugs_by_difficulty(bugs, "all")) == 1

def test_filter_bugs_easy():
    bugs = [
        {"ground_truth": {"is_ambiguous": False}},
        {"ground_truth": {"is_ambiguous": True}},
    ]
    result = filter_bugs_by_difficulty(bugs, "easy")
    assert len(result) == 1

def test_filter_bugs_medium():
    bugs = [
        {"ground_truth": {"severity": 3}},
        {"ground_truth": {"severity": 5}},
    ]
    result = filter_bugs_by_difficulty(bugs, "medium")
    assert len(result) == 1  # only severity 5

def test_filter_bugs_hard():
    bugs = [
        {"ground_truth": {"is_ambiguous": True, "root_cause": "bug"}},
        {"ground_truth": {"is_ambiguous": False, "root_cause": "design"}},
        {"ground_truth": {"is_ambiguous": False, "root_cause": "bug"}},
    ]
    result = filter_bugs_by_difficulty(bugs, "hard")
    assert len(result) == 2  # ambiguous OR non-bug root cause
