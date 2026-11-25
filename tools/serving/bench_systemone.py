#!/usr/bin/env python3
"""Drive /v1/systemone with a fixed short and a fixed long request set.

    bench_systemone.py http://127.0.0.1:8000 result.json [rounds]

Sixteen short and eight long requests, four questions each (a choice, two
nouls, a score), sent at concurrency 1 and 4. Each set runs once untimed,
which is also where the recorded answers come from, then `rounds` times
timed; a line a round gives prompt tokens per second over the set and the
median request latency. The states are slices of this repository's own
docs/*.md, so the sets drift when those files change: compare servers on one
checkout. Every question carries `instructions`, which llama-server requires,
and no prompt exceeds 4,096 tokens, the widest it takes at `-ub 4096`.

    bench_systemone.py --compare a.json b.json

prints how far two result files' answers are apart: every returned number
except `confidence`, which the two servers define differently (llama-server
rescales it against chance), and whether the chosen labels agree.
"""
import concurrent.futures as cf
import glob
import json
import os
import statistics as st
import sys
import time
import urllib.request

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")

def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, pre + "/" + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, pre + "/" + str(i))
    else:
        yield pre, x


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    for name in a["answers"]:
        fa, fb = dict(flat(a["answers"][name])), dict(flat(b["answers"][name]))
        d = sorted(abs(fa[k] - fb[k]) for k in fa
                   if isinstance(fa[k], float) and isinstance(fb.get(k), (int, float))
                   and not k.endswith("confidence"))
        lab = [(fa[k], fb.get(k)) for k in fa if isinstance(fa[k], (str, bool))]
        agree = sum(x == y for x, y in lab)
        print(f"{name}: {len(d)} numbers, median |d| {st.median(d):.2e}, "
              f"p99 {d[int(len(d) * 0.99) - 1]:.2e}, max {d[-1]:.2e}; "
              f"labels agree {agree}/{len(lab)}")


if sys.argv[1] == "--compare":
    compare(sys.argv[2], sys.argv[3])
    sys.exit(0)

url, out = sys.argv[1], sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
corpus = ""
for p in sorted(glob.glob(os.path.join(DOCS, "[A-Z]*.md"))):
    if p.endswith(("BENCHMARKS.md", "CLEF.md")):
        continue
    corpus += open(p).read() + "\n"
QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the passage mainly about?", "criteria": {
        "caching": "The passage is mainly about caching or memory layout.",
        "scheduling": "The passage is mainly about scheduling or admission.",
        "kernels": "The passage is mainly about GPU kernels or arithmetic.",
        "serving": "The passage is mainly about the HTTP API or serving.",
        "other": "The passage is mainly about something else."}},
    "has_numbers": {"type": "noul", "instructions": "Does the passage report measured numbers?"},
    "has_code": {"type": "noul", "instructions": "Does the passage contain source code or shell commands?"},
    "depth": {"type": "score", "instructions": "How much background does the passage assume?", "criteria": ["Introductory", "Intermediate", "Expert only"]},
}
def req(start, chars):
    return {"model": "clef-flash", "state": corpus[start:start + chars], "questions": QUESTIONS}
SETS = {
    "short": [req(7001 * i, 1400 + 70 * i) for i in range(16)],
    "long": [req(15013 * i + 500, 9800 + 150 * i) for i in range(8)],
}
def post(body):
    data = json.dumps(body).encode()
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(
        url + "/v1/systemone", data, {"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    return time.perf_counter() - t, j
def run(reqs, conc):
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(post, reqs))
    wall = time.perf_counter() - t
    toks = sum(j["usage"].get("input_tokens", j["usage"].get("prompt_tokens", 0)) for _, j in res)
    return toks / wall, st.median(l for l, _ in res) * 1e3, res
result = {"runs": [], "answers": {}, "tokens": {}}
for name, reqs in SETS.items():          # warm-up, untimed: both servers settle their buffers
    _, _, res = run(reqs, 1)
    result["answers"][name] = [j["answers"] for _, j in res]
    result["tokens"][name] = [j["usage"].get("input_tokens", j["usage"].get("prompt_tokens", 0)) for _, j in res]
for r in range(rounds):
    for name, reqs in SETS.items():
        for conc in (1, 4):
            tps, med, _ = run(reqs, conc)
            result["runs"].append({"set": name, "conc": conc, "round": r, "tok_s": tps, "median_ms": med})
            print(f"{name:5} c{conc} round {r}: {tps:8.1f} tok/s  median {med:7.1f} ms", flush=True)
for name in SETS:
    t = result["tokens"][name]
    print(f"{name}: {len(t)} requests, {min(t)}-{max(t)} tokens")
json.dump(result, open(out, "w"))
