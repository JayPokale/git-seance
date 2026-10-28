"""Score retrieval and answers on questions whose answering threads are known.

Usage: python eval/run.py DATA_DIR/request__request [--answers] [--model gemma4:e4b]

For each question and each mode (vector, graph):
- found: the retrieved evidence includes at least one thread that answers the question
- cited: the final answer cites one of those threads (with --answers; needs the model)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seance.ask import History, check_citations, gather, seance  # noqa: E402

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--rescore", action="store_true",
                    help="re-check the citations of the saved answers (results-answers.json) without the model")
    a = ap.parse_args()
    saved = {}
    if a.rescore:
        a.answers = True
        prev = json.loads((Path(__file__).parent / "results-answers.json").read_text())
        saved = {(r["question"], r["mode"]): r["answer"] for r in prev["rows"]}
    h = History(Path(a.data))
    totals = {m: {"found": 0, "cited": 0, "removed": 0, "seconds": 0.0} for m in ("vector", "graph")}
    rows = []
    for q in QUESTIONS:
        truth = {f"#{n}" for n in q["threads"]}
        for mode in ("vector", "graph"):
            start = time.time()
            if a.rescore:
                evidence = gather(h, q["question"], mode)
                answer, cited_ids, removed = check_citations(saved[(q["question"], mode)], evidence)
                out = {"answer": answer, "cited": cited_ids, "removed": removed, "evidence": evidence}
                evidence, cited, removed = out["evidence"], set(out["cited"]), out["removed"]
            elif a.answers:
                out = seance(h, q["question"], mode, a.model)
                evidence, cited, removed = out["evidence"], set(out["cited"]), out["removed"]
            else:
                evidence, cited, removed = gather(h, q["question"], mode), set(), []
            found = bool(truth & {e.cite for e in evidence})
            hit = bool(truth & cited)
            t = totals[mode]
            t["found"] += found
            t["cited"] += hit
            t["removed"] += len(removed)
            t["seconds"] += time.time() - start
            rows.append({"question": q["question"], "mode": mode, "found": found, "cited": hit,
                         "cited_ids": sorted(cited), "removed": removed,
                         "answer": out["answer"] if a.answers else None})
            print(f"{mode:6} found={found!s:5} cited={hit!s:5} {q['question'][:70]}", flush=True)
    n = len(QUESTIONS)
    print()
    for mode, t in totals.items():
        line = f"{mode:6} answering thread retrieved: {t['found']}/{n}"
        if a.answers:
            line += f" | cited in answer: {t['cited']}/{n} | invented citations removed: {t['removed']}"
        print(line + f" | {t['seconds']:.0f} s")
    out_path = Path(__file__).parent / ("results-answers.json" if a.answers else "results-retrieval.json")
    out_path.write_text(json.dumps({"totals": totals, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
