"""
Post-processing script to label data automatically based on the keywords.
This script auto-detects the natural distribution of the fetched dataset
without forcing unnatural targets.
"""
import json
import os
import sys

# Allow running from scripts/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keywords import ROOT_CAUSE_KEYWORDS, CRITICAL_KEYWORDS, SEVERITY_LABEL_MAP

# Derived keyword sets for severity re-labeling
SEV5_KW = SEVERITY_LABEL_MAP[5]
SEV4_KW = SEVERITY_LABEL_MAP[4]
SEV2_KW = SEVERITY_LABEL_MAP[2]
SEV1_KW = SEVERITY_LABEL_MAP[1]

with open("data/bugs_processed.json", encoding="utf-8") as f:
    bugs = json.load(f)

# --- Keyword-based auto-detection ---
for bug in bugs:
    gt = bug["ground_truth"]
    labels_lower = [l.lower() for l in bug.get("labels", [])]
    title_lower = bug.get("title", "").lower()
    body_lower = bug.get("body", "").lower()[:1000]
    text = title_lower + " " + " ".join(labels_lower) + " " + body_lower

    # Criticality
    crit_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
    gt["criticality"] = "critical" if crit_hits > 0 else "non_critical"

    # Root cause
    rc_scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in ROOT_CAUSE_KEYWORDS.items()}
    best = max(rc_scores, key=rc_scores.get)
    if rc_scores[best] == 0:
        best = "bug"
    gt["root_cause"] = best

    # Severity by keyword
    if any(kw in text for kw in SEV5_KW):
        gt["severity"] = 5
    elif any(kw in text for kw in SEV4_KW):
        gt["severity"] = 4
    elif any(kw in text for kw in SEV1_KW):
        gt["severity"] = 1
    elif any(kw in text for kw in SEV2_KW):
        gt["severity"] = 2
    else:
        rc = gt["root_cause"]
        ic = gt["criticality"] == "critical"
        sev_map = {"documentation": 1, "external": 2, "design": 3, "environment": 3, "performance": 3, "bug": 4 if ic else 3}
        gt["severity"] = sev_map.get(rc, 3)

    # Ambiguous: strict tie at score >= 2
    top2 = sorted(rc_scores.values(), reverse=True)[:2]
    gt["is_ambiguous"] = (len(top2) == 2 and top2[0] >= 2 and top2[0] == top2[1])

# --- Final stats ---
print(f"Total: {len(bugs)}")
crit = sum(1 for b in bugs if b["ground_truth"]["criticality"] == "critical")
print(f"Critical: {crit} ({crit*100//len(bugs)}%) | Non-critical: {len(bugs)-crit}")
for lvl in range(1, 6):
    n = sum(1 for b in bugs if b["ground_truth"]["severity"] == lvl)
    print(f"  Severity {lvl}: {n}")

rc_counts = {}
for b in bugs:
    rc_counts[b["ground_truth"]["root_cause"]] = rc_counts.get(b["ground_truth"]["root_cause"], 0) + 1
for k in sorted(rc_counts):
    print(f"  {k}: {rc_counts[k]}")

amb = sum(1 for b in bugs if b["ground_truth"]["is_ambiguous"])
print(f"Ambiguous: {amb}")

with open("data/bugs_processed.json", "w", encoding="utf-8") as f:
    json.dump(bugs, f, indent=2, ensure_ascii=False)
print("Saved auto-detected distribution to bugs_processed.json.")
