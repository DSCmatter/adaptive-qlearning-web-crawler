# Phase 6: Evaluation & Baselines Results

## Final Benchmark Run

- **Command**: `python experiments\evaluate_baseline.py --max-pages 10 --runs-per-seed 2 --max-seeds-per-topic 2 --output-prefix PHASE_6_EVAL_STRICT_V2`
- **Seeds evaluated**: `6`
- **Topics**: `blockchain`, `climate_science`, `machine_learning`
- **Crawler variants**: `7`

## Summary Table

| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |
| --- | --- | --- | --- | --- | --- |
| random | 0.133 +/- 0.049 | 0.133 | 0.133 | 0.85 | 22.17 |
| best_first | 0.133 +/- 0.078 | 0.133 | 0.133 | 0.85 | 21.41 |
| pagerank | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.48 | 19.42 |
| pure_q | 1.000 +/- 0.000 | 1.000 | 1.000 | 11.71 | 2.71 |
| pure_bandit | 0.100 +/- 0.000 | 0.100 | 0.100 | 0.43 | 31.23 |
| hybrid_no_gnn | 0.958 +/- 0.144 | 0.958 | 0.958 | 11.17 | 4.31 |
| hybrid | 0.110 +/- 0.020 | 0.110 | 0.110 | 0.57 | 33.24 |

## Findings

1. The strict topic-aware evaluation is discriminative: simple baselines no longer score artificially close to the learned variants.
2. `pure_q` is the strongest configuration in this benchmark, with `1.000` harvest rate and the highest average reward.
3. `hybrid_no_gnn` remains strong at `0.958` harvest rate, suggesting the Q-learning path is carrying most of the current performance.
4. The full `hybrid` configuration underperforms badly, indicating the current online GNN + bandit integration is still a bottleneck rather than a gain.

## Interpretation

Phase 6 succeeded as an evaluation phase. It established:

- a reusable baseline comparison framework
- a stricter, topic-aware relevance protocol
- a clear ranking between crawler variants

The main unresolved issue is no longer the benchmark itself. It is the quality of the full hybrid policy.

## Canonical Artifacts

- `data/results/PHASE_6_EVAL_STRICT_V2.json`
- `data/results/PHASE_6_EVAL_STRICT_V2.md`
- `data/results/PHASE_6_RESULTS.md`
