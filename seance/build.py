"""Build the graph and the vector index from the downloaded history.

Usage: python -m seance.build DATA_DIR/OWNER__REPO

Node tables: Thread (issues and PRs), Comment, Commit.
Relationship tables: REPLIES (Comment -> Thread), MENTIONS (Commit -> Thread),
REFERS (Thread -> Thread, from "#123" and issue links in a thread or its comments),
PART_OF (Commit -> Thread, for the commits of a merged pull request's branch).
Every node gets a nomic-embed-text vector, stored in LanceDB next to the graph tables.
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import lancedb
import pyarrow as pa

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
MAX_CHARS = 2000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def embed(texts: list[str], prefix: str = "search_document: ") -> list[list[float]]:
    """Embed texts with Ollama, in batches, retrying while the server is busy."""
    vectors = []
    for start in range(0, len(texts), 64):
        batch = [prefix + (t or " ")[:MAX_CHARS] for t in texts[start:start + 64]]
        body = json.dumps({"model": EMBED_MODEL, "input": batch}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{OLLAMA}/api/embed", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    vectors.extend(json.load(resp)["embeddings"])
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        print(f"  embedded {min(start + 64, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return vectors


def thread_refs(text: str, repo: str) -> set[int]:
    """Issue and PR numbers referenced as #123 or as a link into the same repository."""
    refs = {int(n) for n in re.findall(r"(?<![\w/&])#(\d{1,6})\b", text)}
    refs |= {int(n) for n in re.findall(rf"github\.com/{re.escape(repo)}/(?:issues|pull)/(\d+)", text)}
    return refs


def main() -> None:
    src = Path(sys.argv[1])
    repo = src.name.replace("__", "/")
    issues = read_jsonl(src / "issues.jsonl")
    comments = read_jsonl(src / "comments.jsonl")
    commits = read_jsonl(src / "commits.jsonl")
    numbers = {t["number"] for t in issues}

    threads = pa.table({
        "number": [t["number"] for t in issues],
        "kind": [t["kind"] for t in issues],
        "title": [t["title"] for t in issues],
        "author": [t["author"] for t in issues],
        "created_at": [t["created_at"] for t in issues],
        "state": [t["state"] for t in issues],
        "labels": [", ".join(t["labels"]) for t in issues],
        "text": [f"{t['title']}\n\n{t['body']}" for t in issues],
        "url": [t["url"] for t in issues],
    })
    comment_table = pa.table({
        "comment_id": [c["id"] for c in comments],
        "number": [c["number"] for c in comments],
        "author": [c["author"] for c in comments],
        "created_at": [c["created_at"] for c in comments],
        "text": [c["body"] for c in comments],
        "url": [c["url"] for c in comments],
    })
    commit_table = pa.table({
        "sha": [c["sha"] for c in commits],
        "author": [c["author"] for c in commits],
        "created_at": [c["date"] for c in commits],
        "text": [c["message"] for c in commits],
        "files": [", ".join(c["files"][:20]) for c in commits],
    })

    replies = pa.table({"comment_id": comment_table["comment_id"], "number": comment_table["number"]})
    mentions = sorted({(c["sha"], n) for c in commits for n in thread_refs(c["message"], repo) if n in numbers})
    refers = set()
    for t in issues:
        refers |= {(t["number"], n) for n in thread_refs(t["title"] + "\n" + t["body"], repo)}
    for c in comments:
        refers |= {(c["number"], n) for n in thread_refs(c["body"], repo)}
    refers = sorted((a, b) for a, b in refers if a != b and b in numbers and a in numbers)

    # Commits in a merged pull request's branch belong to that PR, even when their messages don't mention it.
    part_of = set()
    git = ["git", "-C", str(src / "repo.git")]
    for c in commits:
        m = re.match(r"Merge pull request #(\d+)", c["message"])
        if m and int(m.group(1)) in numbers:
            branch = subprocess.run(git + ["rev-list", "--no-merges", f"{c['sha']}^1..{c['sha']}^2"],
                                    capture_output=True, text=True).stdout.split()
            part_of |= {(sha, int(m.group(1))) for sha in branch}
    part_of = sorted(part_of)

    db = lancedb.connect(src / "lance")
    for name, table in [("Thread", threads), ("Comment", comment_table), ("Commit", commit_table)]:
        if name in db.table_names() and db.open_table(name).count_rows() == table.num_rows:
            print(f"{name}: already embedded, skipping")
            continue
        print(f"{name}: {table.num_rows} rows")
        table = table.append_column("vector", pa.array(embed(table["text"].to_pylist()), pa.list_(pa.float32(), 768)))
        db.create_table(name, table, mode="overwrite")
    db.create_table("REPLIES", replies, mode="overwrite")
    db.create_table("MENTIONS", pa.table({"sha": [m[0] for m in mentions], "number": [m[1] for m in mentions]}), mode="overwrite")
    db.create_table("REFERS", pa.table({"src": [r[0] for r in refers], "dst": [r[1] for r in refers]}), mode="overwrite")
    db.create_table("PART_OF", pa.table({"sha": [p[0] for p in part_of], "number": [p[1] for p in part_of]}), mode="overwrite")
    print(f"REPLIES {replies.num_rows}, MENTIONS {len(mentions)}, REFERS {len(refers)}, PART_OF {len(part_of)}")


if __name__ == "__main__":
    main()
