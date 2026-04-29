"""
GitHub data fetcher — Fetches real bug reports from GitHub repos.

Fetches issues, comments, and contributors from GitHub's REST API,
structures them into the BugTriage schema, labels ground truth using
keyword heuristics, and enriches contributor profiles with expertise data.

Owner: Team Dhurandhar
"""

import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

from src.keywords import (
    REPO_CONFIGS,
    TEAM_MAP,
    CRITICAL_KEYWORDS,
    SEVERITY_LABEL_MAP,
    ROOT_CAUSE_KEYWORDS,
    BUG_DETECTION_LABELS,
    BUG_SKIP_LABELS,
    BUG_TITLE_KEYWORDS,
    BUG_BODY_KEYWORDS,
    BOT_USERNAMES,
)

load_dotenv()

logger = logging.getLogger(__name__)

_rate_lock = threading.Lock()
_last_request_time = 0.0
_MIN_INTERVAL = 0.34


def _build_session(token: str) -> requests.Session:
    """Create a new authenticated session (one per thread)."""
    s = requests.Session()
    if token:
        s.headers["Authorization"] = f"token {token}"
    s.headers["Accept"] = "application/vnd.github.v3+json"
    s.headers["User-Agent"] = "BugTriageEnv/1.0"
    return s


def _rate_limited_get(session: requests.Session, url: str, params: dict = None,
                      retries: int = 3) -> Optional[requests.Response]:
    """Thread-safe GET with global rate limiting and retry logic."""
    global _last_request_time
    for attempt in range(retries):
        with _rate_lock:
            now = time.monotonic()
            gap = _MIN_INTERVAL - (now - _last_request_time)
            if gap > 0:
                time.sleep(gap)
            _last_request_time = time.monotonic()

        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(10, reset - time.time() + 2)
                logger.warning("Rate limited. Waiting %.0fs...", wait)
                time.sleep(wait)
            elif resp.status_code == 422:
                return None  # Validation error, no retry
            else:
                time.sleep(2 ** attempt)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


class GitHubFetcher:
    """Fetches real bug reports from GitHub repositories using REST API.

    Handles issue fetching, comment retrieval, contributor listing,
    ground-truth labeling, and contributor expertise enrichment.
    """

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._session = _build_session(self.token)

    def _get(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Convenience wrapper for rate-limited GET on the main-thread session."""
        return _rate_limited_get(self._session, url, params)

    def fetch_issues(self, repo: str, max_issues: int = 50,
                     state: str = "closed", labels: List[str] = None) -> List[Dict]:
        """Fetch up to *max_issues* closed issues from a repository.

        Args:
            repo: GitHub repo in 'owner/name' format.
            max_issues: Maximum number of issues to return.
            state: Issue state filter ('closed', 'open', 'all').
            labels: Optional label filter.

        Returns:
            List of raw GitHub issue dicts.
        """
        all_issues: List[Dict] = []
        page = 1
        base_url = f"https://api.github.com/repos/{repo}/issues"
        while len(all_issues) < max_issues:
            params = {"state": state, "per_page": 100, "page": page,
                      "sort": "updated", "direction": "desc"}
            if labels:
                params["labels"] = ",".join(labels)
            resp = self._get(base_url, params)
            if not resp:
                break
            issues = resp.json()
            if not isinstance(issues, list) or not issues:
                break
            for issue in issues:
                if issue.get("pull_request"):
                    continue
                all_issues.append(issue)
                if len(all_issues) >= max_issues:
                    break
            if len(issues) < 100:  # last page
                break
            page += 1
        logger.info("Fetched %d raw issues from %s", len(all_issues), repo)
        return all_issues

    def fetch_issue_comments(self, repo: str, issue_number: int,
                             max_comments: int = 3) -> List[str]:
        """Fetch comments for one issue. Thread-safe (creates own session).

        Args:
            repo: GitHub repo in 'owner/name' format.
            issue_number: The issue number to fetch comments for.
            max_comments: Maximum comments to retrieve.

        Returns:
            List of comment body strings (bot comments excluded).
        """
        session = _build_session(self.token)
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        resp = _rate_limited_get(session, url, {"per_page": max_comments})
        if not resp:
            return []
        comments: List[str] = []
        for c in resp.json()[:max_comments]:
            body = (c.get("body") or "").strip()
            user = (c.get("user") or {}).get("login", "")
            if any(bot in user.lower() for bot in BOT_USERNAMES):
                continue
            if body and len(body) > 20:
                comments.append(body[:2000])
        return comments

    def fetch_comments_parallel(self, repo: str, bugs: List[Dict],
                                workers: int = 8) -> List[Dict]:
        """Fetch comments for all bugs in parallel. Modifies bugs in-place.

        Args:
            repo: GitHub repo in 'owner/name' format.
            bugs: List of structured bug dicts to enrich with comments.
            workers: Number of parallel threads.

        Returns:
            The same list of bug dicts, now with 'comments_text' populated.
        """
        def _fetch_one(bug: Dict) -> Dict:
            num = int(bug["bug_id"].split("#")[1])
            bug["comments_text"] = self.fetch_issue_comments(repo, num)
            return bug

        total = len(bugs)
        done = [0]
        lock = threading.Lock()

        def _fetch_tracked(bug: Dict) -> Dict:
            result = _fetch_one(bug)
            with lock:
                done[0] += 1
                if done[0] % 10 == 0:
                    logger.info("Comments: %d/%d", done[0], total)
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_tracked, b) for b in bugs]
            for f in as_completed(futures):
                f.result()  # propagate exceptions
        return bugs

    def fetch_contributors(self, repo: str, max_n: int = 15) -> List[Dict]:
        """Fetch top contributors for a repository.

        Args:
            repo: GitHub repo in 'owner/name' format.
            max_n: Maximum number of contributors.

        Returns:
            List of contributor dicts with 'name' key.
        """
        url = f"https://api.github.com/repos/{repo}/contributors"
        resp = self._get(url, {"per_page": max_n})
        if not resp:
            return []
        return [{"name": c.get("login", "unknown")}
                for c in resp.json()[:max_n]]

    # ------------------------------------------------------------------
    # Bug detection & structuring
    # ------------------------------------------------------------------

    def _is_bug_issue(self, issue: Dict) -> bool:
        """Determine if a raw GitHub issue is a genuine bug report."""
        labels = [lbl.get("name", "").lower() for lbl in issue.get("labels", [])]
        title = (issue.get("title") or "").lower()
        body = (issue.get("body") or "").lower()[:500]

        bug_label = any(
            any(det in lbl for det in BUG_DETECTION_LABELS)
            for lbl in labels
        )
        skip = any(
            any(skip_kw in lbl for skip_kw in BUG_SKIP_LABELS)
            for lbl in labels
        )
        title_bug = any(kw in title for kw in BUG_TITLE_KEYWORDS)
        body_bug = any(kw in body for kw in BUG_BODY_KEYWORDS)
        has_body = len(issue.get("body") or "") > 50

        return (bug_label or title_bug or body_bug) and not skip and has_body

    def process_raw_to_structured(self, raw_issues: List[Dict], repo: str) -> List[Dict]:
        """Convert raw GitHub issues into structured bug dicts."""
        structured: List[Dict] = []
        for issue in raw_issues:
            if not self._is_bug_issue(issue):
                continue
            body = (issue.get("body") or "")[:5000]
            labels = [lbl.get("name", "") for lbl in issue.get("labels", [])]
            closed_by = ""
            if issue.get("closed_by") and isinstance(issue["closed_by"], dict):
                closed_by = issue["closed_by"].get("login", "")
            structured.append({
                "bug_id": f"{repo}#{issue['number']}",
                "title": issue.get("title", ""),
                "body": body,
                "labels": labels,
                "created_at": issue.get("created_at", ""),
                "repo": repo,
                "comments_text": [],
                "author": (issue.get("user") or {}).get("login", "unknown"),
                "is_pull_request": False,
                "_closed_by": closed_by,
            })
        return structured

    def label_ground_truth(self, bugs: List[Dict]) -> List[Dict]:
        """Assign ground-truth labels to each bug using keyword heuristics."""
        for bug in bugs:
            labels_lower = [lbl.lower() for lbl in bug.get("labels", [])]
            title_lower = bug.get("title", "").lower()
            body_lower = bug.get("body", "").lower()[:1000]
            text = f"{title_lower} {' '.join(labels_lower)} {body_lower}"

            crit_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
            crit = "critical" if crit_hits > 0 else "non_critical"

            sev = 3
            for level in [5, 4, 3, 2, 1]:
                if any(kw in text for kw in SEVERITY_LABEL_MAP[level]):
                    sev = level
                    break
            if crit == "critical" and sev < 4:
                sev = 4

            scores = {cat: sum(1 for kw in kws if kw in text)
                      for cat, kws in ROOT_CAUSE_KEYWORDS.items()}
            rc = max(scores, key=scores.get) if max(scores.values()) > 0 else "bug"

            assignee = bug.get("_closed_by") or bug.get("author", "unknown")

            top = sorted(scores.values(), reverse=True)
            is_amb = len(top) >= 2 and top[0] > 0 and top[0] == top[1]

            bug["ground_truth"] = {
                "criticality": crit,
                "severity": sev,
                "root_cause": rc,
                "assignee": assignee,
                "is_ambiguous": is_amb,
            }
            bug.pop("_closed_by", None)
        return bugs

    def _build_contributor_expertise(self, bugs: List[Dict],
                                     contributors_data: Dict) -> Dict:
        """Cross-reference labeled bugs to build expertise profiles per contributor."""
        expertise_map: Dict[str, Dict] = {}
        for bug in bugs:
            assignee = bug.get("ground_truth", {}).get("assignee", "").lower()
            if not assignee or assignee == "unknown":
                continue
            rc = bug["ground_truth"]["root_cause"]
            bug_labels = bug.get("labels", [])

            if assignee not in expertise_map:
                expertise_map[assignee] = {"categories": {}, "labels": set()}

            cat_counts = expertise_map[assignee]["categories"]
            cat_counts[rc] = cat_counts.get(rc, 0) + 1
            expertise_map[assignee]["labels"].update(
                lbl.lower() for lbl in bug_labels
            )

        for _repo, data in contributors_data.items():
            for contrib in data.get("contributors", []):
                name = contrib["name"].lower()
                if name in expertise_map:
                    exp = expertise_map[name]
                    contrib["work_area"] = sorted(
                        exp["categories"],
                        key=exp["categories"].get,
                        reverse=True,
                    )
                    contrib["expertise_labels"] = sorted(exp["labels"])[:15]
                else:
                    contrib["work_area"] = []
                    contrib["expertise_labels"] = []
                contrib.pop("contributions", None)

        return contributors_data

    def save_to_json(self, data, filepath: str) -> None:
        """Save data to a JSON file, creating directories as needed."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        count = len(data) if isinstance(data, list) else "dict"
        logger.info("Saved %s entries to %s", count, filepath)

    def _fetch_repo(self, repo: str, config: Dict, fetch_comments: bool) -> tuple:
        """Fetch one repo's issues + contributors. Designed to run in a thread."""
        target = config.get("max_issues", 50)
        raw = self.fetch_issues(repo, max_issues=target * 5, state="closed")
        structured = self.process_raw_to_structured(raw, repo)[:target]

        if fetch_comments and self.token and structured:
            logger.info("Fetching comments for %d issues from %s...", len(structured), repo)
            self.fetch_comments_parallel(repo, structured, workers=6)

        contribs = self.fetch_contributors(repo)
        contrib_entry = {
            "contributors": contribs,
            "teams": TEAM_MAP.get(repo, ["general"]),
        }
        logger.info("Done %s: %d bugs", repo, len(structured))
        return repo, raw, structured, contrib_entry

    def run_full_pipeline(self, repos: Dict = None, fetch_comments: bool = True,
                          parallel_repos: bool = True) -> List[Dict]:
        """Run the complete data pipeline: fetch, structure, label, enrich, and save."""
        repos = repos or REPO_CONFIGS
        logger.info("=" * 60)
        logger.info("BUG TRIAGE DATA PIPELINE — %d repositories", len(repos))
        logger.info("=" * 60)

        all_raw: List[Dict] = []
        all_structured: List[Dict] = []
        contributors_data: Dict = {}

        if parallel_repos and len(repos) > 1:
            max_workers = min(len(repos), 5)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._fetch_repo, repo, cfg, fetch_comments): repo
                    for repo, cfg in repos.items()
                }
                for future in as_completed(futures):
                    repo, raw, structured, contrib_entry = future.result()
                    all_raw.extend(raw)
                    all_structured.extend(structured)
                    contributors_data[repo] = contrib_entry
        else:
            for repo, config in repos.items():
                logger.info("--- %s ---", repo)
                _, raw, structured, contrib_entry = self._fetch_repo(repo, config, fetch_comments)
                all_raw.extend(raw)
                all_structured.extend(structured)
                contributors_data[repo] = contrib_entry

        self.save_to_json(all_raw, "data/bugs_raw.json")

        labeled = self.label_ground_truth(all_structured)
        self.save_to_json(labeled, "data/bugs_processed.json")

        contributors_data = self._build_contributor_expertise(labeled, contributors_data)
        self.save_to_json(contributors_data, "data/contributors.json")

        self._log_summary(labeled)
        return labeled

    def _log_summary(self, bugs: List[Dict]) -> None:
        """Log pipeline summary stats at INFO level."""
        repos: Dict[str, int] = {}
        for b in bugs:
            repos[b["repo"]] = repos.get(b["repo"], 0) + 1
        logger.info("=" * 60)
        logger.info("TOTAL BUGS: %d", len(bugs))
        for r, c in sorted(repos.items()):
            logger.info("  %s: %d", r, c)
        crit = sum(1 for b in bugs if b["ground_truth"]["criticality"] == "critical")
        logger.info("Critical: %d (%d%%) | Non-critical: %d",
                    crit, crit * 100 // max(len(bugs), 1), len(bugs) - crit)
        for lvl in range(1, 6):
            n = sum(1 for b in bugs if b["ground_truth"]["severity"] == lvl)
            logger.info("  Severity %d: %d", lvl, n)
        rc: Dict[str, int] = {}
        for b in bugs:
            k = b["ground_truth"]["root_cause"]
            rc[k] = rc.get(k, 0) + 1
        for k, v in sorted(rc.items()):
            logger.info("  %s: %d", k, v)
        amb = sum(1 for b in bugs if b["ground_truth"]["is_ambiguous"])
        logger.info("Ambiguous: %d", amb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    fetcher = GitHubFetcher()
    fetcher.run_full_pipeline()
