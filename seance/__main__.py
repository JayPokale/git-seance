"""Hold a seance with a dead project: python -m seance DATA_DIR/OWNER__REPO [--model gemma4:e4b] [--mode graph]"""

import argparse
import random
import sys
import time
from pathlib import Path

from seance.ask import History, seance

DIM, ITAL, AMBER, GREY, RESET = "\033[2m", "\033[3m", "\033[38;5;214m", "\033[38;5;245m", "\033[0m"
CANDLE = [
    "      (    ",
    "     ( )   ",
    "      |    ",
    "    .-'-.  ",
    "    |   |  ",
    "    |   |  ",
    "    '---'  ",
]


def slow(text: str, delay: float = 0.012) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay * (6 if ch in ".," else 1))
    print()


def flicker(seconds: float) -> None:
    flames = ["(", ")", "'", "("]
    end = time.time() + seconds
    while time.time() < end:
        sys.stdout.write(f"\r{AMBER}   {random.choice(flames)}  the spirits are gathering...{RESET}")
        sys.stdout.flush()
        time.sleep(0.15)
    sys.stdout.write("\r" + " " * 50 + "\r")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m seance")
    ap.add_argument("data")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--mode", choices=["graph", "vector"], default="graph")
    a = ap.parse_args()

    h = History(Path(a.data))
    threads = h.tables["Thread"]
    years = sorted(d[:4] for d in threads["created_at"].to_pylist())
    print(AMBER + "\n".join(CANDLE) + RESET)
    slow(f"{ITAL}You light a candle for {h.repo} ({years[0]}-{years[-1]}).{RESET}")
    slow(f"{GREY}{threads.num_rows} threads, {h.tables['Comment'].num_rows} comments and "
         f"{h.tables['Commit'].num_rows} commits stir in the dark. Ask, and they will answer. (empty line to leave){RESET}\n")

    while True:
        try:
            question = input(f"{AMBER}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            question = ""
        if not question:
            slow(f"\n{ITAL}The candle goes out.{RESET}")
            return
        start = time.time()
        flicker(0.8)
        out = seance(h, question, a.mode, a.model)
        print()
        slow(f"{ITAL}{out['answer']}{RESET}")
        cited = [e for e in out["evidence"] if e.cite in out["cited"]]
        seen = set()
        for e in cited:
            if e.cite in seen:
                continue
            seen.add(e.cite)
            print(f"{DIM}   [{e.cite}] {e.kind} by {e.author}, {e.date}  {e.url}{RESET}")
        if out["removed"]:
            print(f"{DIM}   ({len(out['removed'])} citation(s) the model invented were removed: "
                  f"{', '.join(out['removed'])}){RESET}")
        print(f"{DIM}   {time.time() - start:.0f}s{RESET}\n")


if __name__ == "__main__":
    main()
