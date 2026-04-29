"""
Keyword taxonomy for Bug Triage RL Environment.

Single source of truth for every keyword list used in:
  - Ground-truth labeling  (github_fetcher.py)
  - Bug detection filtering (github_fetcher.py)
  - Inference prompts       (inference.py)

Owner: Team Dhurandhar
"""

from typing import Dict, FrozenSet, List, Set

__all__ = [
    "REPO_CONFIGS",
    "TEAM_MAP",
    "CRITICAL_KEYWORDS",
    "SEVERITY_LABEL_MAP",
    "ROOT_CAUSE_KEYWORDS",
    "BUG_DETECTION_LABELS",
    "BUG_SKIP_LABELS",
    "BUG_TITLE_KEYWORDS",
    "BUG_BODY_KEYWORDS",
    "BOT_USERNAMES",
]

# ============================================================
# REPOSITORY CONFIGURATION
# ============================================================

REPO_CONFIGS: Dict[str, dict] = {
    # --- Original 5 ---
    "pytorch/pytorch":            {"max_issues": 35},
    "pallets/flask":              {"max_issues": 35},
    "tiangolo/fastapi":           {"max_issues": 45},
    "numpy/numpy":                {"max_issues": 40},
    "python/cpython":             {"max_issues": 30},

    # --- New 10 ---
    "pandas-dev/pandas":          {"max_issues": 40},
    "scikit-learn/scikit-learn":   {"max_issues": 35},
    "google/jax":                 {"max_issues": 35},
    "huggingface/transformers":   {"max_issues": 40},
    "matplotlib/matplotlib":      {"max_issues": 35},
    "pydantic/pydantic":          {"max_issues": 30},
    "home-assistant/core":        {"max_issues": 35},
    "ansible/ansible":            {"max_issues": 35},
    "scipy/scipy":                {"max_issues": 30},
    "aws/aws-cli":                {"max_issues": 30},
}

TEAM_MAP: Dict[str, List[str]] = {
    # --- Original 5 ---
    "pytorch/pytorch":          ["autograd", "distributed", "jit", "quantization", "mobile", "onnx", "cuda"],
    "pallets/flask":            ["routing", "blueprints", "cli", "testing", "templating", "security"],
    "tiangolo/fastapi":         ["routing", "dependencies", "security", "websockets", "openapi", "middleware"],
    "numpy/numpy":              ["core", "linalg", "fft", "random", "ma", "testing", "io"],
    "python/cpython":           ["stdlib", "interpreter", "gc", "typing", "asyncio", "io", "ssl"],
    # --- New 10 ---
    "pandas-dev/pandas":        ["indexing", "io", "groupby", "dtypes", "plotting", "reshaping", "missing-data"],
    "scikit-learn/scikit-learn": ["classification", "regression", "clustering", "preprocessing", "metrics", "pipeline"],
    "google/jax":               ["compiler", "xla", "autograd", "pmap", "sharding", "random", "linalg"],
    "huggingface/transformers": ["modeling", "tokenization", "training", "generation", "onnx", "quantization"],
    "matplotlib/matplotlib":    ["axes", "figure", "backends", "animation", "widgets", "text", "colors"],
    "pydantic/pydantic":        ["validation", "serialization", "types", "config", "schema", "generics"],
    "home-assistant/core":      ["integrations", "automation", "frontend", "config", "networking", "zwave", "mqtt"],
    "ansible/ansible":          ["modules", "plugins", "networking", "cloud", "cli", "inventory", "vault"],
    "scipy/scipy":              ["optimize", "signal", "sparse", "stats", "integrate", "interpolate", "linalg"],
    "aws/aws-cli":              ["s3", "ec2", "iam", "lambda", "cloudformation", "ecs", "cli"],
}


# ============================================================
# CRITICALITY KEYWORDS
# ============================================================

CRITICAL_KEYWORDS: FrozenSet[str] = frozenset([
    # Crashes & Failures
    "critical", "crash", "segfault", "segmentation fault", "abort",
    "fatal", "panic", "kernel panic", "core dump", "unhandled exception",
    "system failure", "deadlock", "infinite loop", "hang", "freeze",
    # Data Integrity
    "data loss", "dataloss", "data corruption", "corruption", "data destroyed",
    "silent data", "wrong result", "incorrect output", "result mismatch",
    # Security
    "security", "vulnerability", "cve", "exploit", "injection",
    "authentication bypass", "privilege escalation", "xss", "csrf",
    "remote code execution", "rce", "unauthorized access",
    # Priority / Severity Labels
    "p0", "blocker", "showstopper", "release blocker",
    # Regressions
    "regression", "broke", "breaking change",
    # Memory
    "memory corruption", "buffer overflow", "use after free",
    "double free", "nullptr", "null pointer", "out of bounds",
])


# ============================================================
# SEVERITY LABEL MAP  (level -> keyword set)
# ============================================================

SEVERITY_LABEL_MAP: Dict[int, FrozenSet[str]] = {
    5: frozenset([
        "p0", "blocker", "critical", "crash", "security", "segfault",
        "showstopper", "fatal", "data loss", "corruption", "deadlock",
        "release blocker", "vulnerability", "cve", "panic",
        "production outage", "system down",
    ]),
    4: frozenset([
        "p1", "high", "important", "regression", "major",
        "severe", "urgent", "breaking", "blocks", "broken",
        "data integrity", "memory leak", "performance degradation",
        "unusable", "no workaround",
    ]),
    3: frozenset([
        "p2", "medium", "normal", "moderate",
        "intermittent", "sometimes", "edge case", "unexpected behavior",
        "inconsistent", "workaround available", "partial",
    ]),
    2: frozenset([
        "p3", "low", "minor", "small", "inconvenience",
        "cosmetic", "visual", "ui glitch", "style",
        "non-essential", "nice to have", "polish",
    ]),
    1: frozenset([
        "trivial", "typo", "nitpick", "formatting",
        "whitespace", "spelling", "grammar", "comment",
        "documentation only", "docs", "readme",
        "code style", "lint", "cleanup",
    ]),
}


# ============================================================
# ROOT CAUSE KEYWORDS
# ============================================================

ROOT_CAUSE_KEYWORDS: Dict[str, List[str]] = {
    "bug": [
        # Core errors
        "crash", "error", "exception", "traceback", "assertion", "segfault",
        "failure", "broken", "incorrect", "wrong", "unexpected", "raises",
        # Python exception types
        "typeerror", "valueerror", "attributeerror", "indexerror", "keyerror",
        "runtimeerror", "overflowerror", "zerodivisionerror", "lookuperror",
        "stopiteration", "recursionerror", "memoryerror", "oserror",
        "filenotfounderror", "permissionerror", "notimplementederror",
        # Additional patterns
        "null pointer", "nullptr", "none type", "nan", "inf",
        "off-by-one", "race condition", "deadlock", "stack overflow",
        "infinite loop", "silent failure", "data race",
    ],
    "performance": [
        "slow", "memory", "performance", "leak", "speed", "latency",
        "timeout", "resource", "cpu", "oom", "optimization", "bottleneck",
        # Additional patterns
        "memory leak", "high memory", "high cpu", "gpu", "allocation",
        "throughput", "bandwidth", "io bound", "cpu bound",
        "profiling", "benchmark", "regression perf", "cache miss",
        "garbage collection", "gc pressure", "bloat", "scalability",
        "large dataset", "quadratic", "exponential", "n squared",
    ],
    "environment": [
        "config", "install", "setup", "environment", "compatibility",
        "platform", "version", "windows", "linux", "macos", "import", "pip",
        # Additional patterns
        "conda", "virtualenv", "venv", "docker", "container",
        "python version", "python 3", "arm", "arm64", "x86", "aarch64",
        "cuda version", "gpu driver", "nvidia", "rocm",
        "path", "classpath", "dll", "shared library", "so file",
        "locale", "encoding", "utf-8", "ascii", "unicode",
        "ci", "github actions", "jenkins", "tox", "nox",
        "wheel", "sdist", "build", "compile", "cmake", "gcc", "clang",
    ],
    "design": [
        "refactor", "design", "api", "interface", "deprecat", "breaking",
        "inconsist", "behavior", "usability", "confusing", "unintuitive",
        # Additional patterns
        "architecture", "abstraction", "coupling", "cohesion",
        "technical debt", "tech debt", "anti-pattern", "antipattern",
        "naming", "convention", "ergonomic", "footgun", "sharp edge",
        "backward compat", "forward compat", "migration", "upgrade path",
        "public api", "private api", "internal", "exposed",
    ],
    "documentation": [
        "doc", "documentation", "typo", "readme", "example",
        "tutorial", "docstring", "spelling", "misleading",
        # Additional patterns
        "changelog", "release note", "api reference", "guide",
        "getting started", "faq", "how-to", "howto",
        "outdated", "stale", "inaccurate", "unclear", "ambiguous",
        "missing doc", "undocumented", "wrong example",
        "type hint", "annotation", "signature",
    ],
    "external": [
        "upstream", "dependency", "third-party", "external", "vendor",
        "library", "package", "conda", "numpy", "scipy",
        # Additional patterns
        "transitive", "pinned", "version conflict", "incompatible",
        "breaking upstream", "upstream bug", "vendor bug",
        "system library", "os bug", "kernel bug", "glibc",
        "openssl", "libssl", "ssl certificate", "tls",
        "cloud provider", "aws", "gcp", "azure",
        "api change", "deprecated upstream", "eol",
    ],
}


# ============================================================
# BUG DETECTION FILTERS  (used in _is_bug_issue)
# ============================================================

BUG_DETECTION_LABELS: FrozenSet[str] = frozenset([
    "bug", "defect", "error", "fault", "issue",
    "regression", "crash", "broken", "fix", "hotfix",
    "confirmed bug", "verified bug", "accepted",
    "type: bug", "type:bug", "kind/bug", "kind: bug",
    "type: defect", "category: bug",
])

BUG_SKIP_LABELS: FrozenSet[str] = frozenset([
    "feature", "enhancement", "question", "discussion",
    "rfc", "proposal", "wontfix", "won't fix",
    "duplicate", "invalid", "stale", "not a bug",
    "support", "help wanted", "good first issue",
])

BUG_TITLE_KEYWORDS: FrozenSet[str] = frozenset([
    "error", "bug", "crash", "fail", "broken", "incorrect", "wrong",
    "exception", "traceback", "segfault", "regression", "fix",
    "issue", "problem", "unexpected", "cannot", "unable",
    "does not work", "doesn't work", "not working",
    "throws", "raises", "assertion", "panic",
    "corrupt", "leak", "hang", "freeze", "timeout",
])

BUG_BODY_KEYWORDS: FrozenSet[str] = frozenset([
    "traceback", "error", "exception", "crash",
    "stack trace", "stacktrace", "assertion failed",
    "segmentation fault", "core dumped", "panic",
    "expected", "actual", "reproduce", "steps to reproduce",
    "minimal example", "mwe", "mcve",
])


# ============================================================
# BOT FILTER  (comment authors to ignore)
# ============================================================

BOT_USERNAMES: FrozenSet[str] = frozenset([
    "bot", "github-actions", "codecov", "dependabot",
    "renovate", "greenkeeper", "stale", "lgtm",
    "coveralls", "pre-commit-ci", "mergify",
])
