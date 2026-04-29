#!/usr/bin/env python3
"""
Demo Script - Bug Triage RL Environment
Team Dhurandhar | Meta/PyTorch OpenEnv Hackathon

This script demonstrates the environment API using hardcoded heuristics.
Outputs use the exact mandatory [START]/[STEP]/[END] format as required
by the OpenEnv automated judge, with added human-readable formatting.
"""

from typing import List, Optional
from src.env import BugTriageEnv
from src.models import BugTriageAction

def log_start(task: str, env: str, model: str) -> None:
    """Emit the mandatory [START] line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], details: str = "") -> None:
    """Emit the mandatory [STEP] line, then a human-readable summary if provided."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)
    if details:
        print(f"       -> {details}")

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the mandatory [END] line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def print_header(title: str, description: str):
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")
    print(f"{description}\n")

def demo_task_criticality(env: BugTriageEnv) -> float:
    print_header(
        "TASK 1: Criticality Detection (Easy)",
        "Goal: Classify each bug as 'critical' or 'non_critical'.\nScoring: 1.0 = correct, 0.0 = wrong."
    )
    
    scores = []
    all_rewards = []
    log_start("task_criticality", "bug-triage", "heuristic-agent")
    
    for ep in range(5):
        obs = env.reset(task_id="task_criticality")
        bug = obs.bug_report

        text = (bug.title + " " + " ".join(bug.labels)).lower()
        if any(kw in text for kw in ["crash", "segfault", "security", "data loss", "blocker"]):
            prediction = "critical"
        else:
            prediction = "non_critical"

        action = BugTriageAction(
            task_id="task_criticality",
            bug_id=bug.bug_id,
            criticality=prediction,
            confidence=0.85,
            reasoning="Heuristic scan of title and labels",
        )

        result = env.step(action)
        scores.append(result.reward)
        all_rewards.append(result.reward)
        
        gt = result.metadata["info"]["ground_truth"]["criticality"]
        human_readable = f"Bug #{bug.bug_id[:8]} | Pred: {prediction:<12} | Truth: {gt:<12} | Reward: {result.reward:.2f}"
        
        log_step(ep + 1, f"CRITICALITY: {prediction}", result.reward, True, None, human_readable)

    avg = sum(scores) / len(scores)
    log_end(avg >= 0.1, len(scores), avg, all_rewards)
    print(f"\nTask 1 Average Score: {avg:.3f}")
    return avg

def demo_task_severity(env: BugTriageEnv) -> float:
    print_header(
        "TASK 2: Severity Scoring (Medium)",
        "Goal: Assign severity 1-5 (1=trivial, 5=crash).\nScoring: 1.0 exact, 0.7 off-by-1, 0.4 off-by-2."
    )
    
    scores = []
    all_rewards = []
    log_start("task_severity", "bug-triage", "heuristic-agent")
    
    for ep in range(5):
        obs = env.reset(task_id="task_severity")
        bug = obs.bug_report

        text = (bug.title + " " + " ".join(bug.labels)).lower()
        if any(kw in text for kw in ["crash", "segfault", "blocker", "security"]):
            prediction = 5
        elif any(kw in text for kw in ["regression", "broken", "error"]):
            prediction = 4
        elif any(kw in text for kw in ["bug", "wrong", "incorrect"]):
            prediction = 3
        elif any(kw in text for kw in ["minor", "cosmetic", "enhancement"]):
            prediction = 2
        else:
            prediction = 3

        action = BugTriageAction(
            task_id="task_severity",
            bug_id=bug.bug_id,
            severity=prediction,
            confidence=0.7,
            reasoning="Label-based severity estimation",
        )

        result = env.step(action)
        scores.append(result.reward)
        all_rewards.append(result.reward)
        
        gt = result.metadata["info"]["ground_truth"]["severity"]
        human_readable = f"Bug #{bug.bug_id[:8]} | Pred: {prediction} | Truth: {gt} | Reward: {result.reward:.2f}"
        
        log_step(ep + 1, f"SEVERITY: {prediction}", result.reward, True, None, human_readable)

    avg = sum(scores) / len(scores)
    log_end(avg >= 0.1, len(scores), avg, all_rewards)
    print(f"\nTask 2 Average Score: {avg:.3f}")
    return avg

def demo_task_root_cause(env: BugTriageEnv) -> float:
    print_header(
        "TASK 3: Root Cause + Assignee (Hard)",
        "Goal: Identify root cause category AND pick the best assignee.\nScoring: (0.6 * root_cause) + (0.4 * assignee)."
    )
    
    scores = []
    all_rewards = []
    log_start("task_root_cause_assignee", "bug-triage", "heuristic-agent")
    
    for ep in range(5):
        obs = env.reset(task_id="task_root_cause_assignee")
        bug = obs.bug_report

        text = (bug.title + " " + bug.body[:500] + " " + " ".join(bug.labels)).lower()
        if any(kw in text for kw in ["install", "config", "version", "import", "platform"]):
            root_cause = "environment"
        elif any(kw in text for kw in ["crash", "error", "wrong", "null"]):
            root_cause = "bug"
        elif any(kw in text for kw in ["slow", "memory", "performance"]):
            root_cause = "performance"
        elif any(kw in text for kw in ["doc", "typo", "readme"]):
            root_cause = "documentation"
        else:
            root_cause = "environment"

        assignee = obs.available_assignees[0] if obs.available_assignees else "unknown"

        action = BugTriageAction(
            task_id="task_root_cause_assignee",
            bug_id=bug.bug_id,
            root_cause=root_cause,
            assignee=assignee,
            confidence=0.6,
            reasoning="Keyword-based root cause + first assignee heuristic",
        )

        result = env.step(action)
        scores.append(result.reward)
        all_rewards.append(result.reward)
        
        gt_rc = result.metadata["info"]["ground_truth"]["root_cause"]
        human_readable = f"Bug #{bug.bug_id[:8]} | Pred RC: {root_cause:<12} | Truth RC: {gt_rc:<12} | Reward: {result.reward:.2f}"
        
        log_step(ep + 1, f"{root_cause} | ASGN: {assignee}", result.reward, True, None, human_readable)

    avg = sum(scores) / len(scores)
    log_end(avg >= 0.1, len(scores), avg, all_rewards)
    print(f"\nTask 3 Average Score: {avg:.3f}")
    return avg

def main():
    print("=" * 80)
    print("  BUG TRIAGE ENV - DEMO")
    print("\nThis script provides a 5-episode baseline demo using fast heuristics.")
    print("Outputs conform exactly to the OpenEnv stdout specifications.\n")

    env = BugTriageEnv(seed=42)
    score1 = demo_task_criticality(env)
    score2 = demo_task_severity(env)
    score3 = demo_task_root_cause(env)

    overall = (score1 + score2 + score3) / 3
    
    print("\n")
    print(f"  Criticality Detection:  {score1:.3f}")
    print(f"  Severity Scoring:       {score2:.3f}")
    print(f"  Root Cause + Assignee:  {score3:.3f}")
    print(f"  --------------------------------")
    print(f"  OVERALL SCORE:          {overall:.3f}")
    print(f"\n{'=' * 80}")
    print("  DEMO SUMMARY (Average Reward out of 1.0)")
    print("  These scores signify how well the simple hardcoded heuristic")
    print("  agent performed against the human ground truth on these 5 bugs.")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
