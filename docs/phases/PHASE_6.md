# Phase 6: Evaluation & Baselines

**Timeline**: Week 7-8  
**Status**: ✅ Complete  
**Date Started**: April 2, 2026  
**Date Completed**: April 2, 2026

---

## Overview

Phase 6 adds the missing evaluation layer for the project. Instead of treating evaluation as a one-off script, this phase introduces reusable baseline crawler implementations, a shared experiment runner, and report generation utilities so the hybrid crawler can be compared consistently against heuristic and ablation baselines.

---

## Objectives

1. Implement documented baseline crawlers for comparison.
2. Add a reusable evaluation harness that aggregates run metrics.
3. Support both live HTTP crawling and offline testing.
4. Generate machine-readable and markdown reports in `data/results`.

---

## Deliverables Implemented

- `src/crawler/baseline_crawlers.py`
  - `RandomCrawler`
  - `BestFirstCrawler`
  - `PageRankCrawler`
- `src/utils/relevance.py`
  - topic-aware relevance scoring derived from the seed topic files
  - stricter filtering for non-content Wikipedia and Wikimedia URLs
- `src/evaluation/crawler_evaluator.py`
  - shared run normalization
  - mean/std aggregation
  - JSON + markdown report writers
- `experiments/evaluate_baseline.py`
  - compares heuristic baselines plus hybrid ablations
  - writes output to `data/results`
- `tests/test_phase6_evaluation.py`
  - verifies offline evaluation flow using an in-memory mini web

---

## Evaluation Configurations

The evaluation runner currently supports:

1. `random`
2. `best_first`
3. `pagerank`
4. `pure_q`
5. `pure_bandit`
6. `hybrid_no_gnn`
7. `hybrid`

The ablation configurations reuse the existing `AdaptiveCrawler` by toggling trained components on and off, which keeps the comparison aligned with the production hybrid path instead of duplicating crawl logic in multiple places.

---

## How to Run

```bash
.\venv\Scripts\python experiments\evaluate_baseline.py --max-pages 20 --runs-per-seed 1
```

Optional flags:

- `--max-seeds-per-topic`: limit how many seeds to use from each topic file
- `--output-prefix`: customize output filenames

Outputs:

- `data/results/PHASE_6_EVAL.json`
- `data/results/PHASE_6_EVAL.md`
- `data/results/PHASE_6_RESULTS.md`

---

## Metrics Captured

- Harvest Rate
- Precision@10
- Precision@20
- Crawl Time
- Total Reward
- Average Reward
- Total Pages Crawled
- Relevant Pages Found

---

## Notes

- Local tests cover the evaluation framework without needing live network access.
- Live evaluation now scores relevance per seed topic instead of using one broad keyword list for every crawl.
- The PageRank baseline now benefits from dynamic PageRank cache invalidation as the discovered graph changes during crawling.

---

## Final Results

The final Phase 6 benchmark used:

```bash
python experiments\evaluate_baseline.py --max-pages 10 --runs-per-seed 2 --max-seeds-per-topic 2 --output-prefix PHASE_6_EVAL_STRICT_V2
```

This evaluated `6` seeds across `3` topics and compared `7` crawler variants:

| Crawler | Harvest Rate | P@10 | Avg Reward |
| --- | --- | --- | --- |
| random | 0.133 +/- 0.049 | 0.133 | 0.85 |
| best_first | 0.133 +/- 0.078 | 0.133 | 0.85 |
| pagerank | 0.100 +/- 0.000 | 0.100 | 0.48 |
| pure_q | 1.000 +/- 0.000 | 1.000 | 11.71 |
| pure_bandit | 0.100 +/- 0.000 | 0.100 | 0.43 |
| hybrid_no_gnn | 0.958 +/- 0.144 | 0.958 | 11.17 |
| hybrid | 0.110 +/- 0.020 | 0.110 | 0.57 |

Key takeaways:

1. The stricter topic-aware evaluation is now discriminative and useful.
2. `pure_q` is the strongest performing configuration in the current system.
3. `hybrid_no_gnn` performs well, but the full `hybrid` configuration still underperforms badly.
4. Phase 6 therefore succeeded in validating the evaluation framework and exposing a real weakness in the current online hybrid integration.

---

## Next Steps

1. Diagnose why the full `hybrid` crawler performs worse than `pure_q` and `hybrid_no_gnn`.
2. Improve online GNN + bandit integration and rerun the strict benchmark.
3. Add statistical significance testing and ablation analysis summaries if publication-quality reporting is needed.
