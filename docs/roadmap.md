# Roadmap to v0.1

**Written:** 2026-11-02. The runtime is research-ready (experiments 001–007); this is the list that separates
it from a release strangers can install and use. Dates are targets; each item gets an experiment or changelog
entry when it lands.

| # | item | why | estimate | target |
|---|---|---|---|---|
| R1 ✔ | Learned, class-free scorer (experiment 009, done 2026-11-02) | the zero-shot scorer needs class names; closes the last 2.5–4 points to the oracle at 4 frames | 3–4 days | 2026-11-06 |
| R2 ✔ | Millisecond-budget scheduler (experiment 010, done 2026-11-03) | the research contribution: given a budget in ms, choose frames, windows and refinement stride from the measured curves | 3 days | 2026-11-09 |
| R3 ✔ | Generality (011: held-out split, 1080p, HEVC, VP9, keyframe density; 012: Charades, boundary = action coverage) | every number so far is 320×180 H.264 sports video with 20 classes | 1 week | 2026-11-16 |
| R4 ✔ | Robustness (013, done 2026-11-05): `vam.errors`, forkserver pool + scan timeout, degenerate inputs → uniform, scheduler uniform fallback | a library fails in the field on inputs the benchmark never sees | 3 days | 2026-11-19 |
| R5 | Model coverage: two more adapters, one of them a policy / vision-language-action model | the unified-API claim needs both uses proven | 1 week | 2026-11-26 |
| R6 | Packaging: PyPI wheel, pinned dependency ranges, CI running the tests on Linux and Windows | installability | 2 days | 2026-11-28 |
| R7 | Docs: quickstart, API reference, two example notebooks | adoption | 2–3 days | 2026-12-01 |

Deferred past v0.1: NVDEC / torchcodec decode (008). The decode stage is a ~90 ms seek-bound floor that is
already below uniform sampling's latency; hardware decode would add an install burden that costs users more
than it saves.

Progress is tracked in [`CHANGELOG.md`](CHANGELOG.md) and [`experiments/README.md`](experiments/README.md).
