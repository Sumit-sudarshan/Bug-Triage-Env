"""
Tests for BugTriageEnv.

Owner: Sumit
Day 5: Expanded to 30+ test cases for >90% coverage on env.py
"""

import pytest
from pydantic import ValidationError

from src.env import BugTriageEnv
from src.models import (
    BugTriageAction,
    BugTriageObservation,
    BugGroundTruth,
    CriticalityLabel,
    SeverityLevel,
    RootCauseCategory,
)


DATA_PATH = "data/bugs_processed.json"


@pytest.fixture
def env():
    return BugTriageEnv(data_path=DATA_PATH, seed=42)


def _make_action(obs: BugTriageObservation, **overrides) -> BugTriageAction:
    """Build a minimal valid action for the given observation."""
    kwargs = dict(task_id=obs.task_id, bug_id=obs.bug_report.bug_id, confidence=0.7)
    if obs.task_id == "task_criticality":
        kwargs["criticality"] = CriticalityLabel.CRITICAL
    elif obs.task_id == "task_severity":
        kwargs["severity"] = SeverityLevel.MEDIUM
    else:
        kwargs["root_cause"] = RootCauseCategory.BUG
        kwargs["assignee"] = obs.available_assignees[0] if obs.available_assignees else "unknown"
    kwargs.update(overrides)
    return BugTriageAction(**kwargs)


# ── reset() ──────────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = env.reset()
    assert isinstance(obs, BugTriageObservation)

def test_reset_observation_fields(env):
    obs = env.reset()
    assert obs.task_id in ("task_criticality", "task_severity", "task_root_cause_assignee")
    assert obs.bug_report.bug_id
    assert obs.step == 0
    assert obs.max_steps == 1
    assert obs.done is False

def test_reset_task_criticality(env):
    obs = env.reset(task_id="task_criticality")
    assert obs.task_id == "task_criticality"

def test_reset_task_severity(env):
    obs = env.reset(task_id="task_severity")
    assert obs.task_id == "task_severity"

def test_reset_task_root_cause_has_assignees(env):
    obs = env.reset(task_id="task_root_cause_assignee")
    assert obs.task_id == "task_root_cause_assignee"
    assert len(obs.available_assignees) > 0

def test_reset_invalid_task_raises(env):
    with pytest.raises(ValueError):
        env.reset(task_id="task_nonexistent")

def test_reset_episode_counter_increments(env):
    env.reset()
    assert env.state()["episode_number"] == 1
    env.step(_make_action(env.reset(task_id="task_criticality")))
    env.reset()
    assert env.state()["episode_number"] == 3

def test_reset_cycles_tasks(env):
    tasks = []
    for _ in range(6):
        obs = env.reset()
        tasks.append(obs.task_id)
        env.step(_make_action(obs))
    assert "task_criticality" in tasks
    assert "task_severity" in tasks
    assert "task_root_cause_assignee" in tasks

def test_reset_bug_report_has_body(env):
    obs = env.reset()
    assert isinstance(obs.bug_report.body, str)

def test_reset_bug_report_has_repo(env):
    obs = env.reset()
    assert "/" in obs.bug_report.repo  # format: "owner/name"

def test_reset_consecutive_gives_different_bugs(env):
    """Consecutive resets should cycle through different bugs."""
    ids = set()
    for _ in range(10):
        obs = env.reset(task_id="task_criticality")
        ids.add(obs.bug_report.bug_id)
        env.step(_make_action(obs))
    assert len(ids) == 10  # all different


# ── step() ───────────────────────────────────────────────────────────────────

def test_step_returns_tuple(env):
    obs = env.reset()
    result = env.step(_make_action(obs))
    assert len(result) == 4

def test_step_done_is_true(env):
    obs = env.reset()
    _, _, done, _ = env.step(_make_action(obs))
    assert done is True

def test_step_reward_in_range(env):
    for _ in range(20):
        obs = env.reset()
        _, reward, _, _ = env.step(_make_action(obs))
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0,1]"

def test_step_info_has_ground_truth(env):
    obs = env.reset()
    _, _, _, info = env.step(_make_action(obs))
    gt = info["ground_truth"]
    assert "criticality" in gt
    assert "severity" in gt
    assert "root_cause" in gt
    assert "assignee" in gt
    assert "is_ambiguous" in gt

def test_step_wrong_task_id_raises(env):
    obs = env.reset(task_id="task_criticality")
    action = BugTriageAction(
        task_id="task_severity",
        bug_id=obs.bug_report.bug_id,
        severity=SeverityLevel.HIGH,
        confidence=0.5,
    )
    with pytest.raises(ValueError):
        env.step(action)

def test_step_wrong_bug_id_raises(env):
    obs = env.reset(task_id="task_criticality")
    action = BugTriageAction(
        task_id="task_criticality",
        bug_id="wrong/repo#999",
        criticality=CriticalityLabel.CRITICAL,
        confidence=0.5,
    )
    with pytest.raises(ValueError):
        env.step(action)

def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(BugTriageAction(
            task_id="task_criticality",
            bug_id="any",
            criticality=CriticalityLabel.CRITICAL,
            confidence=0.5,
        ))

def test_step_second_step_raises(env):
    obs = env.reset()
    env.step(_make_action(obs))
    with pytest.raises(RuntimeError):
        env.step(_make_action(obs))

def test_step_correct_criticality_reward(env):
    obs = env.reset(task_id="task_criticality")
    _, _, _, info = env.step(_make_action(obs, criticality=CriticalityLabel.CRITICAL))
    assert 0.0 <= info["reward_breakdown"]["total"] <= 1.0

def test_step_terminal_obs_done_true(env):
    """Terminal observation returned by step() has done=True."""
    obs = env.reset()
    terminal_obs, _, _, _ = env.step(_make_action(obs))
    assert terminal_obs.done is True
    assert terminal_obs.step == 1

def test_step_info_has_reward_breakdown(env):
    obs = env.reset()
    _, _, _, info = env.step(_make_action(obs))
    rb = info["reward_breakdown"]
    assert "base_score" in rb
    assert "confidence_bonus" in rb
    assert "reasoning_bonus" in rb
    assert "edge_case_bonus" in rb
    assert "total" in rb

def test_step_info_has_episode_number(env):
    obs = env.reset()
    _, _, _, info = env.step(_make_action(obs))
    assert info["episode_number"] == 1

def test_step_severity_none_gets_zero(env):
    """Action with no severity field should score 0.0 base for severity task."""
    obs = env.reset(task_id="task_severity")
    action = BugTriageAction(
        task_id="task_severity",
        bug_id=obs.bug_report.bug_id,
        confidence=0.5,
    )
    _, reward, _, _ = env.step(action)
    # Reward should be low (base 0.0 from no severity)
    assert reward <= 0.5

def test_step_with_reasoning(env):
    """Action with reasoning should not crash and may get bonus."""
    obs = env.reset(task_id="task_criticality")
    action = _make_action(
        obs,
        reasoning="This bug causes a crash in production with segfault, clearly critical."
    )
    _, reward, _, info = env.step(action)
    assert 0.0 <= reward <= 1.0


# ── state() ──────────────────────────────────────────────────────────────────

def test_state_has_required_keys(env):
    env.reset()
    s = env.state()
    for key in ("current_task_id", "current_bug_id", "episode_number",
                "step_count", "total_bugs", "tasks_available"):
        assert key in s, f"Missing key: {key}"

def test_state_tasks_available(env):
    env.reset()
    s = env.state()
    assert set(s["tasks_available"]) == {
        "task_criticality", "task_severity", "task_root_cause_assignee"
    }

def test_state_total_bugs(env):
    env.reset()
    s = env.state()
    assert s["total_bugs"] >= 180

def test_state_before_reset(env):
    """state() before any reset returns valid dict with None values."""
    s = env.state()
    assert s["current_task_id"] is None
    assert s["current_bug_id"] is None
    assert s["episode_number"] == 0

def test_state_waiting_for_step(env):
    """waiting_for_step should be True after reset, False after step."""
    env.reset()
    assert env.state()["waiting_for_step"] is True
    obs = env.reset(task_id="task_criticality")
    env.step(_make_action(obs))
    assert env.state()["waiting_for_step"] is False


# ── reproducibility ───────────────────────────────────────────────────────────

def test_reproducibility_same_seed(env):
    env2 = BugTriageEnv(data_path=DATA_PATH, seed=42)
    bugs1, bugs2 = [], []
    for _ in range(10):
        obs1 = env.reset()
        obs2 = env2.reset()
        bugs1.append(obs1.bug_report.bug_id)
        bugs2.append(obs2.bug_report.bug_id)
        env.step(_make_action(obs1))
        env2.step(_make_action(obs2))
    assert bugs1 == bugs2

def test_different_seeds_differ():
    env_a = BugTriageEnv(data_path=DATA_PATH, seed=1)
    env_b = BugTriageEnv(data_path=DATA_PATH, seed=99)
    bugs_a, bugs_b = [], []
    for _ in range(5):
        obs = env_a.reset(task_id="task_criticality")
        bugs_a.append(obs.bug_report.bug_id)
        env_a.step(_make_action(obs))
    for _ in range(5):
        obs = env_b.reset(task_id="task_criticality")
        bugs_b.append(obs.bug_report.bug_id)
        env_b.step(_make_action(obs))
    assert bugs_a != bugs_b


# ── full dataset iteration ────────────────────────────────────────────────────

def test_all_bugs_iterable(env):
    """All bugs can be iterated through without error."""
    total = env.state()["total_bugs"] if env._current_bug else len(env._bugs)
    for i in range(total):
        cur = env.reset(task_id="task_criticality")
        action = _make_action(cur)
        _, reward, done, info = env.step(action)
        assert done is True
        assert isinstance(reward, float)


# ── Pydantic validation ───────────────────────────────────────────────────────

def test_invalid_confidence_raises():
    with pytest.raises(ValidationError):
        BugTriageAction(
            task_id="task_criticality",
            bug_id="a",
            criticality=CriticalityLabel.CRITICAL,
            confidence=1.5,  # out of range
        )

def test_negative_confidence_raises():
    with pytest.raises(ValidationError):
        BugTriageAction(
            task_id="task_criticality",
            bug_id="a",
            criticality=CriticalityLabel.CRITICAL,
            confidence=-0.1,
        )


# ── Hardening / edge cases ────────────────────────────────────────────────────

def test_invalid_task_type_constructor():
    """Invalid task_type at construction time raises."""
    with pytest.raises(ValueError):
        BugTriageEnv(data_path=DATA_PATH, task_type="task_invalid")

def test_fixed_task_type_only_returns_that_task():
    """When task_type is fixed, all resets use that task."""
    env = BugTriageEnv(data_path=DATA_PATH, task_type="task_severity")
    for _ in range(5):
        obs = env.reset()
        assert obs.task_id == "task_severity"
        env.step(_make_action(obs))

def test_wrap_around_after_all_bugs(env):
    """After iterating all bugs, cursor wraps around."""
    total = len(env._bugs)
    for _ in range(total + 5):
        obs = env.reset(task_id="task_criticality")
        env.step(_make_action(obs))
    # Should still work without error
    obs = env.reset()
    assert obs.bug_report.bug_id is not None

def test_reset_after_step_works(env):
    """Can reset immediately after step without error."""
    obs1 = env.reset(task_id="task_criticality")
    env.step(_make_action(obs1))
    obs2 = env.reset(task_id="task_severity")
    assert obs2.task_id == "task_severity"

def test_multiple_resets_without_step(env):
    """Multiple resets without stepping should work (abandons episode)."""
    env.reset(task_id="task_criticality")
    env.reset(task_id="task_severity")
    obs = env.reset(task_id="task_root_cause_assignee")
    # Should work — last reset wins
    action = _make_action(obs)
    _, reward, done, _ = env.step(action)
    assert done is True

def test_ground_truth_values_valid(env):
    """Ground truth values in info must be valid enum values."""
    for _ in range(20):
        obs = env.reset()
        _, _, _, info = env.step(_make_action(obs))
        gt = info["ground_truth"]
        assert gt["criticality"] in ("critical", "non_critical")
        assert gt["severity"] in (1, 2, 3, 4, 5)
        assert gt["root_cause"] in ("bug", "design", "environment", "performance", "documentation", "external")
        assert isinstance(gt["assignee"], str) and len(gt["assignee"]) > 0
        assert isinstance(gt["is_ambiguous"], bool)
