# Phase 7: Finalization, Diagnostics, and Project Closure

**Timeline**: Week 9 (fast-track)  
**Status**: Complete  
**Date Started**: April 3, 2026
**Date Completed**: April 4, 2026

---

## Overview

Phase 7 finalized the project with a reproducible strict benchmark protocol, targeted hybrid diagnostics, and an explicit production policy decision backed by evidence.

---

## Phase 7 Objectives

1. Lock a canonical evaluation configuration for apples-to-apples comparisons. ✅
2. Capture per-run evidence for why hybrid diverges from strong variants. ✅
3. Evaluate frozen-policy behavior vs update-enabled behavior. ✅
4. Produce final reproducible reports and closure documentation. ✅

---

## Implemented Kickoff Changes (April 3, 2026)

### 1) Deterministic evaluation controls
Updated `experiments/evaluate_baseline.py` with:
- `--random-seed`
- deterministic per-run seeding across Python, NumPy, and Torch
- `--crawler-filter` for focused ablation runs
- `--seed-url` for exact seed replay

### 2) Debuggable evaluation output
Updated evaluation flow to optionally retain detailed run artifacts:
- `--include-run-details` in `experiments/evaluate_baseline.py`
- `run_details` payload support in `src/evaluation/crawler_evaluator.py`

### 3) Adaptive crawler diagnostics and evaluation mode control
Updated `src/crawler/adaptive_crawler.py` with:
- `enable_diagnostics` flag
- step-level vector stats and decision traces
- `online_updates` toggle for frozen-policy evaluation mode

### 4) Bandit score introspection
Updated `src/models/contextual_bandit.py` with:
- `score_candidates(...)` helper
- shared scoring path used by link selection and diagnostics

---

## Executed Commands

### A) Hybrid diagnosis (strict, trace-enabled)

```bash
python experiments\evaluate_baseline.py \
  --max-pages 10 \
  --runs-per-seed 2 \
  --max-seeds-per-topic 2 \
  --crawler-filter pure_q,hybrid_no_gnn,hybrid \
  --random-seed 42 \
  --include-run-details \
  --enable-diagnostics \
  --output-prefix PHASE_7_DIAG_STRICT
```

### B) Frozen-policy comparison (online updates disabled)

```bash
python experiments\evaluate_baseline.py \
  --max-pages 10 \
  --runs-per-seed 2 \
  --max-seeds-per-topic 2 \
  --crawler-filter pure_q,hybrid_no_gnn,hybrid \
  --random-seed 42 \
  --disable-online-updates \
  --output-prefix PHASE_7_FROZEN_STRICT
```

### C) Final strict benchmark (all available variants)

```bash
python experiments\evaluate_baseline.py \
  --max-pages 10 \
  --runs-per-seed 2 \
  --max-seeds-per-topic 2 \
  --random-seed 42 \
  --output-prefix PHASE_7_FINAL_STRICT
```

---

## Final Metrics

### Diagnostic strict (PHASE_7_DIAG_STRICT)

| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |
| --- | --- | --- | --- | --- | --- |
| pure_q | 0.958 +/- 0.144 | 0.958 | 0.958 | 11.20 | 2.72 |
| hybrid_no_gnn | 1.000 +/- 0.000 | 1.000 | 1.000 | 11.72 | 2.48 |
| hybrid | 0.117 +/- 0.044 | 0.117 | 0.117 | 0.67 | 32.81 |

### Frozen strict (PHASE_7_FROZEN_STRICT)

| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |
| --- | --- | --- | --- | --- | --- |
| pure_q | 0.958 +/- 0.144 | 0.958 | 0.958 | 11.21 | 2.98 |
| hybrid_no_gnn | 1.000 +/- 0.000 | 1.000 | 1.000 | 11.70 | 2.68 |
| hybrid | 0.117 +/- 0.044 | 0.117 | 0.117 | 0.69 | 27.61 |

### Final strict (PHASE_7_FINAL_STRICT)

| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |
| --- | --- | --- | --- | --- | --- |
| random | 0.108 +/- 0.029 | 0.108 | 0.108 | 0.55 | 21.84 |
| best_first | 0.133 +/- 0.078 | 0.133 | 0.133 | 0.84 | 22.50 |
| pagerank | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.48 | 19.06 |
| pure_q | 0.958 +/- 0.144 | 0.958 | 0.958 | 11.21 | 2.77 |
| pure_bandit | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.45 | 27.29 |
| hybrid_no_gnn | 1.000 +/- 0.000 | 1.000 | 1.000 | 11.72 | 2.56 |
| hybrid | 0.117 +/- 0.044 | 0.117 | 0.117 | 0.69 | 25.65 |

---

## Diagnostic Finding

Run-detail analysis from PHASE_7_DIAG_STRICT showed a consistent early divergence in hybrid:

- Hybrid run details observed: 12 runs
- Hybrid runs with a second decision step: 12
- Hybrid second-step irrelevant selections: 12/12
- Selection mode at divergence: bandit with 50 candidates

This confirms the unresolved issue is still in the online hybrid selection path, not in baseline or evaluator plumbing.

---

## Final Policy Decision

- Selected production policy: hybrid_no_gnn
- Reason: highest harvest rate (1.000), highest precision (1.000), strong reward (11.72), and fast crawl time (2.56s)
- Fallback policy: pure_q (0.958 harvest, stable and fast)
- Experimental branch retained: full hybrid (known limitation, not production-ready)

---

## Deliverables


- Reproducible strict benchmark commands and artifacts for diagnosis, frozen comparison, and final ranking.
- Instrumented diagnostic run-details pipeline for per-step behavior analysis.
- Final policy recommendation and closure summary in phase docs.

---

## Canonical Artifacts

- data/results/PHASE_7_DIAG_STRICT.json
- data/results/PHASE_7_DIAG_STRICT.md
- data/results/PHASE_7_FROZEN_STRICT.json
- data/results/PHASE_7_FROZEN_STRICT.md
- data/results/PHASE_7_FINAL_STRICT.json
- data/results/PHASE_7_FINAL_STRICT.md
- data/results/PHASE_7_RESULTS.md

---

## Exit Criteria Check

1. Canonical strict benchmark reproducible: Complete.
2. First divergence evidence captured for hybrid: Complete.
3. Final production policy selected and justified: Complete.
4. Closure documentation and artifacts published: Complete.

Phase 7 is complete.
