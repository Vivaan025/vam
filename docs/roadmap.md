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
| R5 ✔ | Model coverage (014, done 2026-11-05): torchvision R(2+1)D-18 / R3D-18 / MViT-v2-S, `ActionPolicy` + OpenVLA-7B; selection gain holds on the second family, policy runs through `Stream` and the cost model | the unified-API claim needs both uses proven | 1 week | 2026-11-26 |
| R6 ✔ | Packaging (015, done 2026-11-05): `vam-runtime` wheel with the packaged scorer, bounded ranges, GitHub Actions on Linux + Windows | installability | 2 days | 2026-11-28 |
| R7 ✔ | Docs (016, done 2026-11-06): quickstart, API reference, two executed notebooks, `BudgetedSampler` | adoption | 2–3 days | 2026-12-01 |

Deferred past v0.1: NVDEC / torchcodec decode (008). The decode stage is a ~90 ms seek-bound floor that is
already below uniform sampling's latency; hardware decode would add an install burden that costs users more
than it saves.

## After v0.1: the comparison a reviewer asks for

**Written:** 2026-11-07. v0.1.0 is released. Every number so far is a linear probe on 200 THUMOS14 videos;
the adaptive-frame-selection literature reports mAP on ActivityNet-1.3 and top-1 on Mini-Kinetics with a
trained ResNet-50, against each other. These two items put `vam` in that table.

| # | item | why | estimate | target |
|---|---|---|---|---|
| R8 | Standard benchmarks (017): ActivityNet-1.3 val and Mini-Kinetics under the FrameExit protocol (1 fps, ResNet-50, mAP / top-1, GFLOPs), FrameExit checkpoint reproduced first; FCVID once the data request is granted | the benchmark the field recognises | 1-2 weeks (data-bound) | 2026-11-21 |
| R9 | Prior samplers (018): FrameExit, OCSampler, AR-Net from released weights; SCSampler, AdaFrame, SMART re-implemented; one table of mAP x GFLOPs x measured ms on the same machine and classifier | "how does it do against at least two of them under the same backbone and budget" | 2-3 weeks | 2026-12-12 |

Progress is tracked in [`CHANGELOG.md`](CHANGELOG.md) and [`experiments/README.md`](experiments/README.md).
