"""Summarise eval JSONL into an accuracy-vs-latency table, grouped by GPU, sampler and budget.

    python benchmarks/summarize.py results/ucf101.jsonl [--csv out.csv]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    for line in open(args.path):
        r = json.loads(line)
        groups[(r["gpu"], r["backbone"], r["sampler"], r["budget"])].append(r)

    rows = []
    for (gpu, backbone, sampler, budget), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        n = len(rs)
        mean = lambda k: sum(r[k] for r in rs) / n
        rows.append({
            "gpu": gpu, "backbone": backbone, "sampler": sampler, "budget": budget, "n": n,
            "top1": sum(r["correct"] for r in rs) / n,
            "select_ms": mean("select_ms"), "decode_ms": mean("decode_ms"), "infer_ms": mean("infer_ms"),
            "total_ms": mean("select_ms") + mean("decode_ms") + mean("infer_ms"),
            "peak_mb": mean("infer_peak_mb"),
            "hit_rate": mean("hit_rate") if "hit_rate" in rs[0] else float("nan"),
        })

    hdr = f"{'gpu':28} {'sampler':8} {'budget':>6} {'n':>5} {'top1':>6} {'select':>8} {'decode':>8} {'infer':>8} {'total':>8} {'hit':>5}"
    print(hdr)
    for r in rows:
        print(f"{r['gpu'][:28]:28} {r['sampler']:8} {r['budget']:6d} {r['n']:5d} {r['top1']:6.3f} "
              f"{r['select_ms']:8.1f} {r['decode_ms']:8.1f} {r['infer_ms']:8.1f} {r['total_ms']:8.1f} {r['hit_rate']:5.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
