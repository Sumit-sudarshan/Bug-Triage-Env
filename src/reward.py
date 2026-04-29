"""
Reward calculator for Bug Triage environment.

Combines the base grader score with bonus signals to produce a richer
reward that encourages:
  - Confidence calibration (agent's confidence matches actual accuracy)
  - Reasoning quality (longer, thoughtful reasoning on correct answers)
  - Edge case handling (correct classification of ambiguous bugs)

Owner: Team Dhurandhar
"""

import logging
from src.models import BugTriageAction, BugGroundTruth, BugTriageReward

logger = logging.getLogger(__name__)

__all__ = ["RewardCalculator"]


class RewardCalculator:
    """Computes sophisticated rewards beyond raw grader scores."""

    def compute(
        self,
        base_score: float,
        action: BugTriageAction,
        ground_truth: BugGroundTruth,
    ) -> BugTriageReward:
        """Calculate the full reward with all bonus components."""
        confidence_diff = abs(action.confidence - base_score)
        if confidence_diff < 0.15:
            confidence_bonus = 0.08
        elif confidence_diff < 0.3:
            confidence_bonus = 0.02
        else:
            confidence_bonus = -0.05

        reasoning_len = len(action.reasoning.strip())
        if reasoning_len > 50 and base_score > 0.7:
            reasoning_bonus = 0.05
        elif reasoning_len > 100 and base_score < 0.5:
            reasoning_bonus = 0.02
        else:
            reasoning_bonus = 0.0

        if ground_truth.is_ambiguous and base_score == 1.0:
            edge_case_bonus = 0.1
        else:
            edge_case_bonus = 0.0

        total = base_score + confidence_bonus + reasoning_bonus + edge_case_bonus
        total = max(0.0, min(1.0, total))

        return BugTriageReward(
            base_score=base_score,
            confidence_bonus=confidence_bonus,
            reasoning_bonus=reasoning_bonus,
            edge_case_bonus=edge_case_bonus,
            total=total,
        )
