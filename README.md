# git séance 🕯️

Ask a dead open-source project why it did what it did, and get an answer that cites its own history.

`git séance` turns a repository's issues, pull requests, comments and commits into a graph, and answers questions
with a local open-weight model (Gemma 4 through Ollama) that may only speak from that history. Every claim cites an
issue (`[#3142]`) or a commit (`[a1b2c3d]`), and the code removes any citation that isn't in the retrieved evidence.

The first project summoned is [`request`](https://github.com/request/request), the Node.js HTTP library that went
into maintenance mode in 2020: 3,477 issues and pull requests, 12,441 comments and 2,290 commits.

## How it works

1. **Ingest** (`seance/ingest.py`): issues, pull requests and their comments from the GitHub API, commits from a
   blobless clone.
2. **Build** (`seance/build.py`): node tables `Thread`, `Comment`, `Commit` and relationship tables `REPLIES`
   (comment → thread), `MENTIONS` (commit → thread, from "Fixes #123" and similar) and `REFERS` (thread → thread, from
   `#123` and issue links). Every node is embedded with `nomic-embed-text` and stored in
   [LanceDB](https://github.com/lancedb/lancedb).
3. **Ask** (`seance/ask.py`):
   - vector search finds the closest threads, comments and commits;
   - [lance-graph](https://github.com/lance-format/lance-graph) expands them with Cypher: each seed thread's
     discussion, the commits that mention it, and the threads it refers to;
   - everything is ranked by similarity, and Gemma answers from the top evidence only;
   - citations are checked in code against the evidence.

## Run it

```bash
ollama pull gemma4:e4b && ollama pull nomic-embed-text
pip install lancedb lance-graph pyarrow numpy
python -m seance.ingest request/request data/
python -m seance.build data/request__request
python -m seance data/request__request
```

## Evaluation

`eval/questions.json` holds questions about `request` whose answering threads were checked by hand.
`python eval/run.py data/request__request [--answers]` scores plain vector retrieval against graph retrieval.
Results are in [the evaluation section](#results).

## Results

(filled in from `eval/results-*.json`)

## License

MIT
