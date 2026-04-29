#!/usr/bin/env python3
"""
RL Training script for Bug Triage Env using GRPO.

Trains a language model using Group Relative Policy Optimization with
rewards from BugTriageEnv. Demonstrates that a small RL-trained model
can outperform a larger zero-shot model on bug triage tasks.

Usage:
    python train_rl.py                                          # defaults
    python train_rl.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python train_rl.py --task task_criticality --epochs 3

Owner: Team Dhurandhar
"""

import os
import re
import json
import argparse
import logging
import time
import torch
from datetime import datetime
from typing import Dict

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.env import BugTriageEnv
from src.models import BugTriageAction

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# DEVICE DETECTION
# ============================================================
def get_device_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple MPS (Metal)"
    return "cpu", "CPU"

DEVICE, DEVICE_NAME = get_device_info()
logger.info(f"Device: {DEVICE_NAME}")


# ============================================================
# PROMPT TEMPLATES
# ============================================================
TASK_PROMPTS = {
    "task_criticality": (
        "Classify this bug as critical or non_critical.\n\n"
        "Title: {title}\nBody: {body}\nLabels: {labels}\n\n"
        'Reply JSON: {{"classification": "critical" or "non_critical", '
        '"confidence": 0.0-1.0, "reasoning": "brief"}}'
    ),
    "task_severity": (
        "Assign severity 1-5 (1=trivial, 5=critical crash).\n\n"
        "Title: {title}\nBody: {body}\nLabels: {labels}\n\n"
        'Reply JSON: {{"score": 1-5, "confidence": 0.0-1.0, "reasoning": "brief"}}'
    ),
    "task_root_cause_assignee": (
        "Classify root cause (bug/design/environment/performance/documentation/external) "
        "and pick assignee.\n\n"
        "Title: {title}\nBody: {body}\nLabels: {labels}\n"
        "Assignees: {assignees}\n\n"
        'Reply JSON: {{"root_cause": "category", "assignee": "username", '
        '"confidence": 0.0-1.0, "reasoning": "brief"}}'
    ),
}


# ============================================================
# DATASET BUILDER
# ============================================================
def build_dataset(task_id: str, num_samples: int = 530) -> Dataset:
    """Build a HuggingFace Dataset of prompts from the environment."""
    env = BugTriageEnv(seed=42)
    prompts = []
    metadata = []

    for _ in range(num_samples):
        obs = env.reset(task_id=task_id)
        bug = obs.bug_report

        prompt = TASK_PROMPTS[task_id].format(
            title=bug.title,
            body=bug.body[:800],
            labels=", ".join(bug.labels),
            assignees=", ".join(obs.available_assignees[:10]) if obs.available_assignees else "N/A",
        )
        prompts.append(prompt)
        metadata.append({
            "bug_id": bug.bug_id,
            "task_id": task_id,
            "available_assignees": obs.available_assignees,
        })

        # Step with dummy action to advance env cursor
        dummy = BugTriageAction(
            task_id=task_id, bug_id=bug.bug_id,
            criticality="non_critical" if task_id == "task_criticality" else None,
            severity=3 if task_id == "task_severity" else None,
            root_cause="environment" if task_id == "task_root_cause_assignee" else None,
            assignee=obs.available_assignees[0] if obs.available_assignees else "unknown",
            confidence=0.5, reasoning="",
        )
        env.step(dummy)

    return Dataset.from_dict({"prompt": prompts, "metadata": [json.dumps(m) for m in metadata]})


# ============================================================
# REWARD FUNCTION (connects GRPO to BugTriageEnv)
# ============================================================
def make_reward_fn(task_id: str):
    """Create a reward function that uses BugTriageEnv to score completions."""
    env = BugTriageEnv(seed=123)

    def reward_fn(completions: list, prompts: list = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # Extract text from completion
            if isinstance(completion, list):
                text = "".join(
                    t.get("content", "") if isinstance(t, dict) else str(t)
                    for t in completion
                )
            else:
                text = str(completion)

            obs = env.reset(task_id=task_id)
            action = _parse_response(task_id, text, obs)
            _, reward, _, _ = env.step(action)
            rewards.append(reward)

        return rewards

    return reward_fn


def _parse_response(task_id: str, text: str, obs) -> BugTriageAction:
    """Parse model output into a BugTriageAction."""
    bug = obs.bug_report

    parsed = {}
    try:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (ValueError, TypeError):
        confidence = 0.5
    reasoning = str(parsed.get("reasoning", ""))[:200]

    if task_id == "task_criticality":
        classification = parsed.get("classification", "non_critical")
        if classification not in ("critical", "non_critical"):
            classification = "non_critical"
        return BugTriageAction(
            task_id=task_id, bug_id=bug.bug_id,
            criticality=classification,
            confidence=confidence, reasoning=reasoning,
        )
    elif task_id == "task_severity":
        try:
            score = max(1, min(5, int(parsed.get("score", 3))))
        except (ValueError, TypeError):
            score = 3
        return BugTriageAction(
            task_id=task_id, bug_id=bug.bug_id,
            severity=score,
            confidence=confidence, reasoning=reasoning,
        )
    else:
        root_cause = parsed.get("root_cause", "environment")
        valid_causes = {"bug", "design", "environment", "performance", "documentation", "external"}
        if root_cause not in valid_causes:
            root_cause = "environment"
        assignee = parsed.get("assignee", "")
        if not assignee or assignee not in obs.available_assignees:
            assignee = obs.available_assignees[0] if obs.available_assignees else "unknown"
        return BugTriageAction(
            task_id=task_id, bug_id=bug.bug_id,
            root_cause=root_cause, assignee=assignee,
            confidence=confidence, reasoning=reasoning,
        )


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, tokenizer, task_id: str, num_episodes: int = 30) -> Dict:
    """Evaluate model on task and return metrics."""
    env = BugTriageEnv(seed=999)
    total_reward = 0.0
    correct = 0
    results = []

    model.eval()
    for _ in range(num_episodes):
        obs = env.reset(task_id=task_id)
        bug = obs.bug_report

        prompt = TASK_PROMPTS[task_id].format(
            title=bug.title,
            body=bug.body[:800],
            labels=", ".join(bug.labels),
            assignees=", ".join(obs.available_assignees[:10]) if obs.available_assignees else "N/A",
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        action = _parse_response(task_id, response_text, obs)
        _, reward, _, _ = env.step(action)

        total_reward += reward
        if reward >= 0.9:
            correct += 1
        results.append(reward)

    avg = total_reward / num_episodes
    return {
        "avg_reward": avg,
        "accuracy": correct / num_episodes,
        "min": min(results),
        "max": max(results),
    }


# ============================================================
# COMPARISON REPORT
# ============================================================
def print_comparison_report(
    model_name: str,
    pre_scores: Dict[str, Dict],
    post_scores: Dict[str, Dict],
    training_time: float,
):
    """Print a formatted before/after comparison report."""
    print("\n" + "=" * 70)
    print("  BUG TRIAGE ENV — RL TRAINING REPORT")
    print("  Team Dhurandhar")
    print("=" * 70)
    print(f"\n  Model:          {model_name}")
    print(f"  Device:         {DEVICE_NAME}")
    print(f"  Training time:  {training_time / 60:.1f} minutes")
    print(f"  Date:           {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    print("\n" + "-" * 70)
    print(f"  {'Task':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 70)

    pre_overall = []
    post_overall = []

    for task_id in pre_scores:
        pre = pre_scores[task_id]["avg_reward"]
        post = post_scores[task_id]["avg_reward"]
        delta = post - pre
        sign = "+" if delta >= 0 else ""
        pre_overall.append(pre)
        post_overall.append(post)

        task_name = task_id.replace("task_", "").replace("_", " ").title()
        print(f"  {task_name:<30} {pre:>9.3f}  {post:>9.3f}  {sign}{delta:>8.3f}")

    pre_avg = sum(pre_overall) / len(pre_overall)
    post_avg = sum(post_overall) / len(post_overall)
    delta_avg = post_avg - pre_avg
    sign = "+" if delta_avg >= 0 else ""

    print("-" * 70)
    print(f"  {'OVERALL':<30} {pre_avg:>9.3f}  {post_avg:>9.3f}  {sign}{delta_avg:>8.3f}")
    print("=" * 70)

    if delta_avg > 0:
        pct = (delta_avg / pre_avg) * 100 if pre_avg > 0 else 0
        print(f"\n  RL training improved overall score by {pct:.1f}%")
    print()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RL Training for Bug Triage Env (GRPO)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name")
    parser.add_argument("--task", type=str, default="all",
                        choices=["task_criticality", "task_severity", "task_root_cause_assignee", "all"],
                        help="Task to train on (default: all)")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per task")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models/bug-triage-rl", help="Save directory")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Episodes per evaluation")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--num_samples", type=int, default=100, help="Training samples per task")
    args = parser.parse_args()

    start_time = time.time()

    tasks = (
        ["task_criticality", "task_severity", "task_root_cause_assignee"]
        if args.task == "all" else [args.task]
    )

    # ── Load tokenizer ────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Pre-training evaluation ───────────────────────────────
    logger.info("\nPre-training evaluation...")
    pre_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    if DEVICE == "mps":
        pre_model = pre_model.to("mps")

    pre_scores = {}
    for task_id in tasks:
        pre_scores[task_id] = evaluate_model(pre_model, tokenizer, task_id, args.eval_episodes)
        logger.info(f"[PRE-TRAIN] {task_id}: {pre_scores[task_id]['avg_reward']:.3f}")
    del pre_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── LoRA config ───────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Train each task ───────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    for task_id in tasks:
        task_name = task_id.replace("task_", "").replace("_", " ").title()
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training: {task_name}")
        logger.info(f"{'=' * 60}")

        # Build dataset
        logger.info(f"Building dataset ({args.num_samples} samples)...")
        dataset = build_dataset(task_id, args.num_samples)

        # Reward function
        reward_fn = make_reward_fn(task_id)

        # GRPO config
        task_output_dir = os.path.join(args.output_dir, task_id)
        grpo_config = GRPOConfig(
            output_dir=task_output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_completion_length=100,
            num_generations=4,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            bf16=False,
            fp16=(DEVICE != "cpu"),
            seed=42,
            gradient_accumulation_steps=2,
            log_completions=True,
            remove_unused_columns=False,
        )

        # Initialize trainer
        trainer = GRPOTrainer(
            model=args.model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        # Train
        logger.info("Starting GRPO training...")
        trainer.train()

        # Save
        trainer.save_model(os.path.join(args.output_dir, f"{task_id}_final"))
        logger.info(f"Model saved: {task_id}")

    # ── Post-training evaluation ──────────────────────────────
    logger.info("\nPost-training evaluation...")
    post_scores = {}

    # Load the last trained model for evaluation
    post_model = trainer.model
    post_model.eval()

    for task_id in tasks:
        post_scores[task_id] = evaluate_model(post_model, tokenizer, task_id, args.eval_episodes)
        logger.info(f"[POST-TRAIN] {task_id}: {post_scores[task_id]['avg_reward']:.3f}")

    # ── Report ────────────────────────────────────────────────
    training_time = time.time() - start_time
    print_comparison_report(args.model, pre_scores, post_scores, training_time)

    # Save report
    report = {
        "model": args.model,
        "device": DEVICE_NAME,
        "tasks": tasks,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "training_time_seconds": training_time,
        "pre_training_scores": {t: s["avg_reward"] for t, s in pre_scores.items()},
        "post_training_scores": {t: s["avg_reward"] for t, s in post_scores.items()},
        "improvement": {
            t: post_scores[t]["avg_reward"] - pre_scores[t]["avg_reward"] for t in tasks
        },
    }
    report_path = os.path.join(args.output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved: {report_path}")
    logger.info(f"Total time: {training_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
