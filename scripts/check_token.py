from dotenv import load_dotenv
import os, requests

load_dotenv(override=True)
t = os.environ.get("GITHUB_TOKEN", "")
print(f"Prefix: {t[:4]}... Length: {len(t)}")

r = requests.get("https://api.github.com/rate_limit",
                  headers={"Authorization": f"token {t}"})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    core = d["resources"]["core"]
    print(f"Rate limit: {core['remaining']}/{core['limit']}")
else:
    print(f"Auth failed. Response: {r.text[:200]}")
