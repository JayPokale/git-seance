"""Ask a dead project's history a question, and get an answer that cites it.

Usage: python -m seance.ask DATA_DIR/OWNER__REPO "why was X done?" [--mode graph|vector] [--model gemma4:e4b]

1. The question is embedded and the closest threads, comments and commits are found (vector search).
2. In graph mode, lance-graph expands those seeds with Cypher: each seed thread's discussion, the commits that
   mention it, and the threads it refers to. Everything is ranked by similarity to the question.
3. A local Gemma answers from that evidence only, citing #123 or a commit hash for every claim.
4. Citations are checked in code: one that is not in the evidence is removed and reported.
"""

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lance_graph import CypherQuery, GraphConfig

from seance.build import OLLAMA, embed

SILENCE = 0.45  # below this best similarity, nothing in the history is about the question


@dataclass
class Evidence:
    cite: str  # "#123" or a 7-character commit hash
    kind: str  # thread, comment, commit
    date: str
    author: str
    text: str
    score: float
    url: str = ""


def graph_config() -> GraphConfig:
    return (
        GraphConfig.builder()
        .with_node_label("Thread", "number")
        .with_node_label("Comment", "comment_id")
        .with_node_label("Commit", "sha")
        .with_relationship("REPLIES", "comment_id", "number")
        .with_relationship("MENTIONS", "sha", "number")
        .with_relationship("REFERS", "src", "dst")
        .with_relationship("PART_OF", "sha", "number")
        .build()
    )


class History:
    def __init__(self, src: Path):
        self.db = lancedb.connect(src / "lance")
        self.repo = src.name.replace("__", "/")
        self.tables = {name: self.db.open_table(name).to_arrow() for name in
                       ["Thread", "Comment", "Commit", "REPLIES", "MENTIONS", "REFERS", "PART_OF"]}
        self.config = graph_config()
        threads = self.tables["Thread"]
        self.thread_vec = {n: v for n, v in zip(threads["number"].to_pylist(), threads["vector"].to_numpy(zero_copy_only=False))}
        self.thread_row = {n: i for i, n in enumerate(threads["number"].to_pylist())}

    def commit_url(self, sha: str) -> str:
        return f"https://github.com/{self.repo}/commit/{sha}"

    def cypher(self, query: str) -> pa.Table:
        return CypherQuery(query).with_config(self.config).execute(self.tables)

    def nearest(self, name: str, qvec: list[float], k: int) -> list[dict]:
        rows = self.db.open_table(name).search(qvec).metric("cosine").limit(k).to_list()
        for r in rows:
            r["score"] = 1 - r.pop("_distance")
        return rows


def cosine(qvec: np.ndarray, vectors) -> np.ndarray:
    m = np.stack(vectors)
    return m @ qvec / (np.linalg.norm(m, axis=1) * np.linalg.norm(qvec) + 1e-9)


def gather(h: History, question: str, mode: str, budget: int = 16) -> list[Evidence]:
    qvec = embed([question], prefix="search_query: ")[0]
    if mode == "vector":
        return vector_evidence(h, qvec, budget)
    return graph_evidence(h, qvec, budget)


def vector_evidence(h: History, qvec: list[float], budget: int) -> list[Evidence]:
    """Plain RAG: the nearest threads, comments and commits, ranked by similarity."""
    ev = [Evidence(f"#{r['number']}", "thread", r["created_at"][:10], r["author"], r["text"], r["score"], r["url"])
          for r in h.nearest("Thread", qvec, 8)]
    ev += [Evidence(f"#{r['number']}", "comment", r["created_at"][:10], r["author"], r["text"], r["score"], r["url"])
           for r in h.nearest("Comment", qvec, 8)]
    ev += [Evidence(r["sha"][:7], "commit", r["created_at"][:10], r["author"], r["text"], r["score"], h.commit_url(r["sha"]))
           for r in h.nearest("Commit", qvec, 6)]
    return sorted(ev, key=lambda e: e.score, reverse=True)[:budget]


def graph_evidence(h: History, qvec: list[float], budget: int, graph_slots: int = 6) -> list[Evidence]:
    """GraphRAG on top of vector search, with the same budget. The graph ranks whole threads by everything that
    points at them (the thread, its comments, and the commits that mention or merged it) and, for the best-supported
    threads, adds what vector search tends to miss: the opening post, and the commit that closed or merged it."""
    q = np.asarray(qvec, dtype=np.float32)
    base = vector_evidence(h, qvec, budget - graph_slots)
    support: dict[int, list[float]] = {}
    for r in h.nearest("Thread", qvec, 8):
        support.setdefault(r["number"], []).append(r["score"])
    for r in h.nearest("Comment", qvec, 15):
        support.setdefault(r["number"], []).append(r["score"])
    commit_hits = {r["sha"]: r["score"] for r in h.nearest("Commit", qvec, 8)}
    if commit_hits:
        shas = ", ".join(f"'{sha}'" for sha in commit_hits)
        for rel in ("MENTIONS", "PART_OF"):
            for row in h.cypher(f"MATCH (m:Commit)-[:{rel}]->(t:Thread) WHERE m.sha IN [{shas}] "
                                "RETURN m.sha AS sha, t.number AS number").to_pylist():
                support.setdefault(row["number"], []).append(commit_hits[row["sha"]])
    # Several independent matches count for more than one strong one.
    rank = {n: sum(sorted(s, reverse=True)[:3]) for n, s in support.items()}
    chosen = sorted(rank, key=rank.get, reverse=True)

    have = {(e.kind, e.cite) for e in base}
    threads = h.tables["Thread"]
    added: list[Evidence] = []
    for n in chosen:
        if len(added) >= graph_slots:
            break
        if ("thread", f"#{n}") not in have:
            i = h.thread_row[n]
            added.append(Evidence(f"#{n}", "thread", threads["created_at"][i].as_py()[:10], threads["author"][i].as_py(),
                                  threads["text"][i].as_py(), float(cosine(q, [h.thread_vec[n]])[0]),
                                  threads["url"][i].as_py()))
            have.add(("thread", f"#{n}"))
        for rel in ("MENTIONS", "PART_OF"):
            fixes = h.cypher(
                f"MATCH (m:Commit)-[:{rel}]->(t:Thread) WHERE t.number IN [{n}] "
                "RETURN m.sha AS sha, m.author AS author, m.created_at AS created_at, m.text AS text, m.vector AS vector")
            if not fixes.num_rows:
                continue
            scores = cosine(q, fixes["vector"].to_numpy(zero_copy_only=False))
            row = fixes.drop_columns(["vector"]).to_pylist()[int(np.argmax(scores))]
            if ("commit", row["sha"][:7]) not in have and len(added) < graph_slots:
                added.append(Evidence(row["sha"][:7], "commit", row["created_at"][:10], row["author"], row["text"],
                                      float(scores.max()), h.commit_url(row["sha"])))
                have.add(("commit", row["sha"][:7]))
            break
    return base + added


PROMPT = """You are the ghost of the open-source project {repo}, speaking at a seance. You only know what is written
in the evidence below: issues, pull requests, comments and commits from the project's history.

Answer the question in 3 to 6 sentences, in a calm, slightly eerie voice, as the project remembering its own past.
Every factual claim must end with the citation of the evidence it comes from, exactly as written in brackets,
e.g. [#123] or [a1b2c3d]. Quote short phrases from the evidence where they help. If the evidence does not answer
the question, say that the spirits are silent on this, and do not guess.

Question: {question}

Evidence:
{evidence}
"""


def generate(model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["response"].strip()


def normalize_cite(token: str) -> str | None:
    """'#123', '\\#123', '123' -> '#123'; a 7-40 character hex hash -> its first 7 characters."""
    token = token.strip().lstrip("\\").lstrip("#").strip()
    if token.isdigit():
        return f"#{token}"
    if re.fullmatch(r"[0-9a-f]{7,40}", token):
        return token[:7]
    return None


def check_citations(answer: str, evidence: list[Evidence]) -> tuple[str, list[str], list[str]]:
    """Keep citations that point at the evidence; remove and report the rest. Accepts the forms models write:
    [#123], [\\#123], [123], [a1b2c3d] and lists such as [#12, #34]."""
    allowed = {e.cite for e in evidence}
    kept, removed = [], []

    def fix(m: re.Match) -> str:
        cites = [normalize_cite(t) for t in m.group(1).split(",")]
        if any(c is None for c in cites):
            return m.group(0)  # not a citation, e.g. [x] in prose
        good = [c for c in cites if c in allowed]
        kept.extend(good)
        removed.extend(c for c in cites if c not in allowed)
        return "".join(f"[{c}]" for c in good)

    cleaned = re.sub(r"\[([^\[\]]{1,80})\]", fix, answer)
    return cleaned, sorted(set(kept)), sorted(set(removed))


def seance(h: History, question: str, mode: str = "graph", model: str = "gemma4:e4b") -> dict:
    evidence = gather(h, question, mode)
    if not evidence or evidence[0].score < SILENCE:
        return {"answer": "The spirits are silent on this.", "cited": [], "removed": [], "evidence": evidence}
    lines = [f"[{e.cite}] ({e.kind}, {e.date}, {e.author}): {' '.join(e.text.split())[:600]}" for e in evidence]
    raw = generate(model, PROMPT.format(repo=h.repo, question=question, evidence="\n".join(lines)))
    answer, cited, removed = check_citations(raw, evidence)
    return {"answer": answer, "cited": cited, "removed": removed, "evidence": evidence}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["graph", "vector"], default="graph")
    ap.add_argument("--model", default="gemma4:e4b")
    a = ap.parse_args()
    h = History(Path(a.data))
    out = seance(h, a.question, a.mode, a.model)
    print("\n🕯️  " + out["answer"] + "\n")
    for e in out["evidence"]:
        if e.cite in out["cited"]:
            print(f"   [{e.cite}] {e.kind}, {e.date}, {e.author}")
    if out["removed"]:
        print(f"\n   (removed citations not found in the history: {', '.join(out['removed'])})", file=sys.stderr)


if __name__ == "__main__":
    main()
