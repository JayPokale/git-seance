"""Download a repository's history: issues, pull requests, their comments, and commits.

Usage: python -m seance.ingest OWNER/REPO DATA_DIR
Writes issues.jsonl, comments.jsonl and commits.jsonl to DATA_DIR/OWNER__REPO/.
"""

import json
import subprocess
import sys
from pathlib import Path


def gh_paginate(path: str):
    """Yield every item of a paginated GitHub REST endpoint."""
    out = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        check=True, capture_output=True, text=True,
    ).stdout
    for page in json.loads(out):
        yield from page


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def issues(repo: str):
    for it in gh_paginate(f"repos/{repo}/issues?state=all&per_page=100"):
        yield {
            "number": it["number"],
            "kind": "pr" if "pull_request" in it else "issue",
            "title": it["title"],
            "body": it["body"] or "",
            "author": (it["user"] or {}).get("login", "ghost"),
            "created_at": it["created_at"],
            "closed_at": it["closed_at"],
            "state": it["state"],
            "labels": [label["name"] for label in it["labels"]],
            "url": it["html_url"],
        }


def comments(repo: str):
    for c in gh_paginate(f"repos/{repo}/issues/comments?per_page=100"):
        yield {
            "id": c["id"],
            "number": int(c["issue_url"].rsplit("/", 1)[1]),
            "author": (c["user"] or {}).get("login", "ghost"),
            "created_at": c["created_at"],
            "body": c["body"] or "",
            "url": c["html_url"],
        }


def commits(repo: str, clone_dir: Path):
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "-q", "--bare", "--filter=blob:none", f"https://github.com/{repo}", str(clone_dir)],
            check=True,
        )
    sep = "\x1e"
    log = subprocess.run(
        ["git", "-C", str(clone_dir), "log", "--all", "--name-only",
         f"--format={sep}%H%x1f%an%x1f%aI%x1f%B%x1f"],
        check=True, capture_output=True, text=True,
    ).stdout
    for record in log.split(sep)[1:]:
        sha, author, date, message, files = record.split("\x1f", 4)
        yield {
            "sha": sha,
            "author": author,
            "date": date,
            "message": message.strip(),
            "files": [f for f in files.strip().splitlines() if f],
        }


def main() -> None:
    repo, data_dir = sys.argv[1], Path(sys.argv[2])
    out = data_dir / repo.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    print("commits", write_jsonl(out / "commits.jsonl", commits(repo, out / "repo.git")), flush=True)
    print("issues", write_jsonl(out / "issues.jsonl", issues(repo)), flush=True)
    print("comments", write_jsonl(out / "comments.jsonl", comments(repo)), flush=True)


if __name__ == "__main__":
    main()
