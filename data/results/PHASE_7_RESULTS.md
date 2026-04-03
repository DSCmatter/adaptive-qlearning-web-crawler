# Phase 7: Finalization, Diagnostics, and Project Closure Results

## Completion Date

- Date completed: April 4, 2026
- Status: Complete

## Commands Executed

1. Hybrid diagnosis strict run
- python experiments\evaluate_baseline.py --max-pages 10 --runs-per-seed 2 --max-seeds-per-topic 2 --crawler-filter pure_q,hybrid_no_gnn,hybrid --random-seed 42 --include-run-details --enable-diagnostics --output-prefix PHASE_7_DIAG_STRICT

2. Frozen-policy strict run
- python experiments\evaluate_baseline.py --max-pages 10 --runs-per-seed 2 --max-seeds-per-topic 2 --crawler-filter pure_q,hybrid_no_gnn,hybrid --random-seed 42 --disable-online-updates --output-prefix PHASE_7_FROZEN_STRICT

3. Final strict run across all crawler variants
- python experiments\evaluate_baseline.py --max-pages 10 --runs-per-seed 2 --max-seeds-per-topic 2 --random-seed 42 --output-prefix PHASE_7_FINAL_STRICT

## Final Strict Benchmark Summary

| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |
| --- | --- | --- | --- | --- | --- |
| random | 0.108 +/- 0.029 | 0.108 | 0.108 | 0.55 | 21.84 |
| best_first | 0.133 +/- 0.078 | 0.133 | 0.133 | 0.84 | 22.50 |
| pagerank | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.48 | 19.06 |
| pure_q | 0.958 +/- 0.144 | 0.958 | 0.958 | 11.21 | 2.77 |
| pure_bandit | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.45 | 27.29 |
| hybrid_no_gnn | 1.000 +/- 0.000 | 1.000 | 1.000 | 11.72 | 2.56 |
| hybrid | 0.117 +/- 0.044 | 0.117 | 0.117 | 0.69 | 25.65 |

## Key Diagnostic Finding

Using PHASE_7_DIAG_STRICT run_details:
- hybrid runs: 12
- hybrid runs reaching second decision step: 12
- second-step irrelevant selections in hybrid: 12/12
- second-step selection mode: bandit (candidate_count=50)

This confirms the remaining weakness is the full online hybrid selection path.

## Final Decision

- Production policy: hybrid_no_gnn
- Fallback policy: pure_q
- Full hybrid status: experimental and not production-ready

## Canonical Artifacts

- data/results/PHASE_7_DIAG_STRICT.json
- data/results/PHASE_7_DIAG_STRICT.md
- data/results/PHASE_7_FROZEN_STRICT.json
- data/results/PHASE_7_FROZEN_STRICT.md
- data/results/PHASE_7_FINAL_STRICT.json
- data/results/PHASE_7_FINAL_STRICT.md
- data/results/PHASE_7_RESULTS.md
