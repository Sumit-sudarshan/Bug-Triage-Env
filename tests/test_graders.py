"""
Tests for grader functions.

Owner: Sumit
Day 5: Expanded to 25+ test cases for edge cases and >90% coverage.
"""

import pytest
from src.models import (
    BugTriageAction,
    BugGroundTruth,
    CriticalityLabel,
    SeverityLevel,
    RootCauseCategory,
    BugTriageReward,
)

from src.graders import grade_criticality, grade_severity, grade_root_cause_assignee
from src.reward import RewardCalculator


def _gt(**kwargs) -> BugGroundTruth:
    defaults = dict(
        bug_id="test-bug",
        criticality=CriticalityLabel.CRITICAL,
        severity=SeverityLevel.HIGH,
        root_cause=RootCauseCategory.BUG,
        assignee="dev_abc",
        is_ambiguous=False,
    )
    defaults.update(kwargs)
    return BugGroundTruth(**defaults)


def _action(task_id="task_criticality", **kwargs) -> BugTriageAction:
    defaults = dict(task_id=task_id, bug_id="test-bug", confidence=0.5)
    defaults.update(kwargs)
    return BugTriageAction(**defaults)


# ── grade_criticality ────────────────────────────────────────────────────────

def test_criticality_correct():
    gt = _gt(criticality=CriticalityLabel.CRITICAL)
    a = _action(criticality=CriticalityLabel.CRITICAL)
    assert grade_criticality(a, gt) == 1.0

def test_criticality_incorrect():
    gt = _gt(criticality=CriticalityLabel.CRITICAL)
    a = _action(criticality=CriticalityLabel.NON_CRITICAL)
    assert grade_criticality(a, gt) == 0.0

def test_criticality_non_critical_correct():
    gt = _gt(criticality=CriticalityLabel.NON_CRITICAL)
    a = _action(criticality=CriticalityLabel.NON_CRITICAL)
    assert grade_criticality(a, gt) == 1.0

def test_criticality_non_critical_wrong():
    gt = _gt(criticality=CriticalityLabel.NON_CRITICAL)
    a = _action(criticality=CriticalityLabel.CRITICAL)
    assert grade_criticality(a, gt) == 0.0

def test_criticality_deterministic():
    gt = _gt(criticality=CriticalityLabel.CRITICAL)
    a = _action(criticality=CriticalityLabel.CRITICAL)
    scores = {grade_criticality(a, gt) for _ in range(5)}
    assert len(scores) == 1

def test_criticality_none_action():
    """Action with criticality=None should return 0.0."""
    gt = _gt(criticality=CriticalityLabel.CRITICAL)
    a = _action()  # no criticality field
    assert grade_criticality(a, gt) == 0.0


# ── grade_severity ────────────────────────────────────────────────────────────

def test_severity_exact_match():
    gt = _gt(severity=SeverityLevel.HIGH)
    a = _action(task_id="task_severity", severity=SeverityLevel.HIGH)
    assert grade_severity(a, gt) == 1.0

def test_severity_off_by_one():
    gt = _gt(severity=SeverityLevel.HIGH)  # 4
    a = _action(task_id="task_severity", severity=SeverityLevel.MEDIUM)  # 3
    assert grade_severity(a, gt) == 0.7

def test_severity_off_by_two():
    gt = _gt(severity=SeverityLevel.HIGH)  # 4
    a = _action(task_id="task_severity", severity=SeverityLevel.LOW)  # 2
    assert grade_severity(a, gt) == 0.4

def test_severity_off_by_three_plus():
    gt = _gt(severity=SeverityLevel.CRITICAL)  # 5
    a = _action(task_id="task_severity", severity=SeverityLevel.LOW)  # 2
    assert grade_severity(a, gt) == 0.0

def test_severity_none_action():
    gt = _gt(severity=SeverityLevel.HIGH)
    a = _action(task_id="task_severity")  # no severity
    assert grade_severity(a, gt) == 0.0

def test_severity_deterministic():
    gt = _gt(severity=SeverityLevel.MEDIUM)
    a = _action(task_id="task_severity", severity=SeverityLevel.HIGH)
    scores = {grade_severity(a, gt) for _ in range(5)}
    assert len(scores) == 1

def test_severity_boundary_trivial_critical():
    gt = _gt(severity=SeverityLevel.TRIVIAL)  # 1
    a = _action(task_id="task_severity", severity=SeverityLevel.CRITICAL)  # 5
    assert grade_severity(a, gt) == 0.0  # diff=4, off by 3+

def test_severity_all_exact_matches():
    """Every severity level matched exactly should return 1.0."""
    for sev in SeverityLevel:
        gt = _gt(severity=sev)
        a = _action(task_id="task_severity", severity=sev)
        assert grade_severity(a, gt) == 1.0, f"Failed for {sev}"

def test_severity_off_by_one_both_directions():
    """Off-by-one should be 0.7 whether above or below."""
    gt = _gt(severity=SeverityLevel.MEDIUM)  # 3
    a_up = _action(task_id="task_severity", severity=SeverityLevel.HIGH)  # 4
    a_down = _action(task_id="task_severity", severity=SeverityLevel.LOW)  # 2
    assert grade_severity(a_up, gt) == 0.7
    assert grade_severity(a_down, gt) == 0.7


# ── grade_root_cause_assignee ─────────────────────────────────────────────────

def test_root_cause_assignee_both_correct():
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee",
                root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    score = grade_root_cause_assignee(a, gt)
    assert score == pytest.approx(1.0)

def test_root_cause_correct_assignee_wrong():
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee",
                root_cause=RootCauseCategory.BUG, assignee="dev_xyz")
    score = grade_root_cause_assignee(a, gt)
    assert 0.5 <= score <= 1.0

def test_root_cause_wrong_assignee_wrong():
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee",
                root_cause=RootCauseCategory.PERFORMANCE, assignee="dev_xyz")
    score = grade_root_cause_assignee(a, gt)
    assert 0.0 <= score < 1.0

def test_root_cause_deterministic():
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee",
                root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    scores = {grade_root_cause_assignee(a, gt) for _ in range(5)}
    assert len(scores) == 1

def test_root_cause_score_in_range():
    gt = _gt(root_cause=RootCauseCategory.DESIGN, assignee="user1")
    for rc in RootCauseCategory:
        a = _action(task_id="task_root_cause_assignee", root_cause=rc, assignee="user1")
        score = grade_root_cause_assignee(a, gt)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for root_cause={rc}"

def test_root_cause_none_action():
    """Action with root_cause=None should get 0 for root_cause component."""
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee", assignee="dev_abc")
    score = grade_root_cause_assignee(a, gt)
    # root_cause_score = 0, assignee_score = 1.0 → 0.6*0 + 0.4*1.0 = 0.4
    assert 0.0 <= score <= 0.5

def test_root_cause_assignee_none():
    """Action with assignee=None should get 0 for assignee component."""
    gt = _gt(root_cause=RootCauseCategory.BUG, assignee="dev_abc")
    a = _action(task_id="task_root_cause_assignee", root_cause=RootCauseCategory.BUG)
    score = grade_root_cause_assignee(a, gt)
    # root_cause_score = 1.0, assignee_score = 0 → 0.6*1.0 + 0.4*0 = 0.6
    assert 0.5 <= score <= 0.7


# ── RewardCalculator ──────────────────────────────────────────────────────────

def test_reward_calculator_basic():
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=0.5)
    reward = rc.compute(1.0, a, gt)
    assert isinstance(reward, BugTriageReward)
    assert 0.0 <= reward.total <= 1.0

def test_reward_total_clamped():
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=0.9)
    reward = rc.compute(1.0, a, gt)
    assert reward.total <= 1.0

def test_reward_total_non_negative():
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.NON_CRITICAL, confidence=0.1)
    reward = rc.compute(0.0, a, gt)
    assert reward.total >= 0.0

def test_reward_deterministic():
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=0.7)
    rewards = {rc.compute(1.0, a, gt).total for _ in range(5)}
    assert len(rewards) == 1

def test_reward_confidence_well_calibrated():
    """Well-calibrated confidence (close to base_score) should get positive bonus."""
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=0.95)
    reward = rc.compute(1.0, a, gt)
    assert reward.confidence_bonus > 0

def test_reward_confidence_badly_calibrated():
    """Badly calibrated confidence should get negative bonus."""
    rc = RewardCalculator()
    gt = _gt()
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=0.1)
    reward = rc.compute(1.0, a, gt)
    assert reward.confidence_bonus < 0

def test_reward_edge_case_bonus_ambiguous():
    """Ambiguous bug with perfect score should get edge case bonus."""
    rc = RewardCalculator()
    gt = _gt(is_ambiguous=True)
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=1.0)
    reward = rc.compute(1.0, a, gt)
    assert reward.edge_case_bonus > 0

def test_reward_edge_case_no_bonus_non_ambiguous():
    """Non-ambiguous bug should get zero edge case bonus."""
    rc = RewardCalculator()
    gt = _gt(is_ambiguous=False)
    a = _action(criticality=CriticalityLabel.CRITICAL, confidence=1.0)
    reward = rc.compute(1.0, a, gt)
    assert reward.edge_case_bonus == 0.0
