"""
Utility functions for Bug Triage environment.
"""

import json
import os
from typing import List, Dict, Optional


def load_json(filepath: str):
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text to maximum length."""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def filter_bugs_by_difficulty(bugs: List[Dict], difficulty: str) -> List[Dict]:
    """Filter bugs by difficulty based on ground truth characteristics.

    Easy = criticality task (binary)
    Medium = severity task (5-level)
    Hard = root_cause + assignee (multi-field)
    """
    if difficulty == "easy":
        return [b for b in bugs if not b.get("ground_truth", {}).get("is_ambiguous", False)]
    elif difficulty == "medium":
        return [b for b in bugs if b.get("ground_truth", {}).get("severity", 3) in [1, 2, 4, 5]]
    elif difficulty == "hard":
        return [b for b in bugs if b.get("ground_truth", {}).get("is_ambiguous", False)
                or b.get("ground_truth", {}).get("root_cause") != "bug"]
    return bugs


def get_bug_by_id(bugs: List[Dict], bug_id: str) -> Optional[Dict]:
    """Get a single bug by ID."""
    for bug in bugs:
        if bug.get("bug_id") == bug_id:
            return bug
    return None


def validate_bug_entry(bug: Dict) -> List[str]:
    """Validate a bug entry against the expected schema. Returns list of errors."""
    errors = []
    required = ["bug_id", "title", "body", "labels", "created_at", "repo", "author"]
    for field in required:
        if field not in bug:
            errors.append(f"Missing field: {field}")

    gt = bug.get("ground_truth")
    if not gt:
        errors.append("Missing ground_truth")
    else:
        gt_fields = ["criticality", "severity", "root_cause", "assignee"]
        for f in gt_fields:
            if f not in gt:
                errors.append(f"Missing ground_truth.{f}")
        if gt.get("criticality") not in ("critical", "non_critical"):
            errors.append(f"Invalid criticality: {gt.get('criticality')}")
        if gt.get("severity") not in (1, 2, 3, 4, 5):
            errors.append(f"Invalid severity: {gt.get('severity')}")
        valid_rc = {"bug", "design", "environment", "performance", "documentation", "external"}
        if gt.get("root_cause") not in valid_rc:
            errors.append(f"Invalid root_cause: {gt.get('root_cause')}")
    return errors


def validate_dataset(bugs: List[Dict]) -> Dict:
    """Validate entire dataset and return summary."""
    total = len(bugs)
    valid = 0
    all_errors = []
    for bug in bugs:
        errs = validate_bug_entry(bug)
        if not errs:
            valid += 1
        else:
            all_errors.append({"bug_id": bug.get("bug_id", "unknown"), "errors": errs})
    return {"total": total, "valid": valid, "invalid": total - valid, "errors": all_errors}


def get_data_distribution(bugs: List[Dict]) -> Dict:
    """Get distribution stats for the dataset."""
    stats = {"total": len(bugs), "repos": {}, "criticality": {}, "severity": {},
             "root_cause": {}, "ambiguous": 0}
    for b in bugs:
        r = b.get("repo", "unknown")
        stats["repos"][r] = stats["repos"].get(r, 0) + 1
        gt = b.get("ground_truth", {})
        c = gt.get("criticality", "unknown")
        stats["criticality"][c] = stats["criticality"].get(c, 0) + 1
        s = gt.get("severity", 0)
        stats["severity"][s] = stats["severity"].get(s, 0) + 1
        rc = gt.get("root_cause", "unknown")
        stats["root_cause"][rc] = stats["root_cause"].get(rc, 0) + 1
        if gt.get("is_ambiguous"):
            stats["ambiguous"] += 1
    return stats
