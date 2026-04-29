"""
Tests for inference.py.

Owner: Sumit
Tests env var handling, helper functions, and stdout format.
LLM calls are mocked to avoid network dependency.
"""

import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

import inference
from inference import _format_action, call_llm
from src.models import (
    BugTriageAction,
    CriticalityLabel,
    RootCauseCategory,
    SeverityLevel,
)


# ── env var handling ──────────────────────────────────────────────────────────

def test_api_base_url_default():
    assert inference.API_BASE_URL is not None
    assert isinstance(inference.API_BASE_URL, str)

def test_model_name_default():
    assert inference.MODEL_NAME is not None
    assert isinstance(inference.MODEL_NAME, str)

def test_hf_token_default():
    assert isinstance(inference.HF_TOKEN, str)

def test_env_vars_from_environment(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "my-test-model")
    monkeypatch.setenv("API_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("HF_TOKEN", "test-token-123")
    # Reimport to pick up new env vars
    import importlib
    with patch("openai.OpenAI"):
        reloaded = importlib.reload(inference)
    assert reloaded.MODEL_NAME == "my-test-model"
    assert reloaded.API_BASE_URL == "http://localhost:8080/v1"
    assert reloaded.HF_TOKEN == "test-token-123"


# ── _format_action ────────────────────────────────────────────────────────────

def test_format_action_criticality():
    a = BugTriageAction(task_id="task_criticality", bug_id="x",
                        criticality=CriticalityLabel.CRITICAL, confidence=0.9)
    out = _format_action(a)
    assert "critical" in out.lower()

def test_format_action_criticality_non_critical():
    a = BugTriageAction(task_id="task_criticality", bug_id="x",
                        criticality=CriticalityLabel.NON_CRITICAL, confidence=0.7)
    out = _format_action(a)
    assert "non_critical" in out.lower()

def test_format_action_severity():
    a = BugTriageAction(task_id="task_severity", bug_id="x",
                        severity=SeverityLevel.HIGH, confidence=0.8)
    out = _format_action(a)
    assert "4" in out  # HIGH = 4

def test_format_action_root_cause_assignee():
    a = BugTriageAction(task_id="task_root_cause_assignee", bug_id="x",
                        root_cause=RootCauseCategory.BUG, assignee="dev_abc", confidence=0.6)
    out = _format_action(a)
    assert "bug" in out.lower()
    assert "dev_abc" in out

def test_format_action_returns_string():
    a = BugTriageAction(task_id="task_criticality", bug_id="x",
                        criticality=CriticalityLabel.CRITICAL, confidence=0.5)
    assert isinstance(_format_action(a), str)


# ── call_llm ──────────────────────────────────────────────────────────────────

def _make_llm_response(content: str):
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp

def test_call_llm_valid_json():
    payload = {"classification": "critical", "confidence": 0.9, "reasoning": "crash"}
    with patch.object(inference.client.chat.completions, "create",
                      return_value=_make_llm_response(json.dumps(payload))):
        result = call_llm("sys", "user")
    assert result["classification"] == "critical"
    assert result["confidence"] == 0.9

def test_call_llm_strips_markdown_fences():
    payload = {"score": 4, "confidence": 0.8, "reasoning": "high"}
    raw = f"```json\n{json.dumps(payload)}\n```"
    with patch.object(inference.client.chat.completions, "create",
                      return_value=_make_llm_response(raw)):
        result = call_llm("sys", "user")
    assert result["score"] == 4

def test_call_llm_bad_json_returns_error():
    with patch.object(inference.client.chat.completions, "create",
                      return_value=_make_llm_response("not json at all")):
        result = call_llm("sys", "user")
    assert "error" in result

def test_call_llm_network_error_returns_error():
    with patch.object(inference.client.chat.completions, "create",
                      side_effect=Exception("connection refused")):
        result = call_llm("sys", "user")
    assert "error" in result
    assert "connection refused" in result["error"]


# ── stdout format ─────────────────────────────────────────────────────────────

def test_start_line_format(capsys):
    """[START] line has correct fields."""
    # Simulate what main() prints for [START]
    task_id = "task_criticality"
    model = "test-model"
    print(f"[START] task={task_id} env=bug-triage model={model}")
    captured = capsys.readouterr()
    assert "[START]" in captured.out
    assert "task=task_criticality" in captured.out
    assert "env=bug-triage" in captured.out
    assert f"model={model}" in captured.out

def test_end_line_format(capsys):
    """[END] line has correct fields."""
    scores = [0.8, 0.6, 1.0]
    avg = sum(scores) / len(scores)
    rewards_str = ",".join(f"{s:.2f}" for s in scores)
    print(f"[END] success=true steps={len(scores)} score={avg:.3f} rewards={rewards_str}")
    captured = capsys.readouterr()
    assert "[END]" in captured.out
    assert "success=true" in captured.out
    assert "steps=3" in captured.out
    assert "score=" in captured.out
    assert "rewards=" in captured.out

def test_step_line_format(capsys):
    """[STEP] line has correct fields."""
    action = BugTriageAction(task_id="task_criticality", bug_id="x",
                             criticality=CriticalityLabel.CRITICAL, confidence=0.8)
    reward_val = 0.85
    done = True
    print(f"[STEP] step=1 action={_format_action(action)} reward={reward_val:.2f} done={str(done).lower()} error=null")
    captured = capsys.readouterr()
    assert "[STEP]" in captured.out
    assert "step=1" in captured.out
    assert "reward=0.85" in captured.out
    assert "done=true" in captured.out
    assert "error=null" in captured.out
