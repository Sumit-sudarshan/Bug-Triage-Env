"""Run the full data pipeline."""
import os
import sys
import logging
from dotenv import load_dotenv

# Force reload .env to pick up latest token
load_dotenv(override=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.github_fetcher import GitHubFetcher, _build_session


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("ERROR: No GITHUB_TOKEN in .env")
        sys.exit(1)

    print(f"Token loaded ({len(token)} chars)")

    # Check rate limit using a temp session
    session = _build_session(token)
    r = session.get("https://api.github.com/rate_limit")
    if r.status_code == 200:
        d = r.json()
        remaining = d["resources"]["core"]["remaining"]
        limit = d["resources"]["core"]["limit"]
        print(f"Rate limit: {remaining}/{limit}")
    else:
        print(f"Rate limit check failed: {r.status_code}")
        sys.exit(1)

    fetcher = GitHubFetcher(token=token)
    fetcher.run_full_pipeline(fetch_comments=True)


if __name__ == "__main__":
    main()
