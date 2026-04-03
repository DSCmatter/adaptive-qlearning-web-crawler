"""Shared evaluation helpers for Phase 6 crawler comparisons."""

from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import json
import statistics
from typing import Callable, Dict, Iterable, List, Optional, Sequence

try:
    from utils.metrics import precision_at_k
except ImportError:  # pragma: no cover - fallback for package-style imports
    from src.utils.metrics import precision_at_k


@dataclass
class CrawlerSpec:
    """Factory wrapper for a crawler configuration."""

    name: str
    factory: Callable[..., object]


@dataclass
class CrawlerRunResult:
    """Normalized metrics for a single crawl run."""

    crawler_name: str
    seed_url: str
    run_index: int
    total_crawled: int
    relevant_found: int
    harvest_rate: float
    precision_at_10: float
    precision_at_20: float
    crawl_time: float
    total_reward: float
    avg_reward: float


class CrawlerEvaluator:
    """Evaluate multiple crawler configurations over shared seeds."""

    def __init__(
        self,
        crawler_specs: Sequence[CrawlerSpec],
        seed_urls: Sequence[str],
        runs_per_seed: int = 1,
        before_run: Optional[Callable[[str, str, int], None]] = None,
        include_run_details: bool = False,
    ):
        self.crawler_specs = list(crawler_specs)
        self.seed_urls = list(seed_urls)
        self.runs_per_seed = runs_per_seed
        self.before_run = before_run
        self.include_run_details = include_run_details

    def _normalize_result(
        self,
        crawler_name: str,
        seed_url: str,
        run_index: int,
        raw_result: Dict[str, object]
    ) -> CrawlerRunResult:
        page_history = list(raw_result.get('page_history', []))
        total_crawled = int(raw_result.get('total_crawled', len(page_history)))
        relevant_found = int(raw_result.get('relevant_found', sum(page_history)))

        return CrawlerRunResult(
            crawler_name=crawler_name,
            seed_url=seed_url,
            run_index=run_index,
            total_crawled=total_crawled,
            relevant_found=relevant_found,
            harvest_rate=float(
                raw_result.get(
                    'harvest_rate',
                    relevant_found / max(1, total_crawled)
                )
            ),
            precision_at_10=float(
                raw_result.get(
                    'precision_at_10',
                    precision_at_k(page_history, min(10, total_crawled))
                )
            ),
            precision_at_20=float(
                raw_result.get(
                    'precision_at_20',
                    precision_at_k(page_history, min(20, total_crawled))
                )
            ),
            crawl_time=float(raw_result.get('crawl_time', 0.0)),
            total_reward=float(raw_result.get('total_reward', 0.0)),
            avg_reward=float(
                raw_result.get(
                    'avg_reward',
                    float(raw_result.get('total_reward', 0.0)) / max(1, total_crawled)
                )
            ),
        )

    @staticmethod
    def _mean_std(values: Iterable[float]) -> Dict[str, float]:
        values = list(values)
        if not values:
            return {'mean': 0.0, 'std': 0.0}
        if len(values) == 1:
            return {'mean': float(values[0]), 'std': 0.0}
        return {
            'mean': float(statistics.mean(values)),
            'std': float(statistics.stdev(values)),
        }

    @staticmethod
    def _build_crawler(spec: CrawlerSpec, seed_url: str):
        """Build a crawler, optionally passing the seed into the factory."""
        signature = inspect.signature(spec.factory)
        if len(signature.parameters) == 0:
            return spec.factory()
        return spec.factory(seed_url)

    def evaluate(self) -> Dict[str, object]:
        """Run every crawler over every configured seed."""
        runs: List[CrawlerRunResult] = []
        run_details: List[Dict[str, object]] = []

        for spec in self.crawler_specs:
            for seed_url in self.seed_urls:
                for run_index in range(self.runs_per_seed):
                    if self.before_run is not None:
                        self.before_run(spec.name, seed_url, run_index)
                    crawler = self._build_crawler(spec, seed_url)
                    raw_result = crawler.crawl(seed_url)
                    runs.append(
                        self._normalize_result(
                            crawler_name=spec.name,
                            seed_url=seed_url,
                            run_index=run_index,
                            raw_result=raw_result,
                        )
                    )

                    if self.include_run_details:
                        run_details.append({
                            'crawler_name': spec.name,
                            'seed_url': seed_url,
                            'run_index': run_index,
                            'trace': list(raw_result.get('trace', [])),
                            'diagnostics': list(raw_result.get('diagnostics', [])),
                            'page_history': list(raw_result.get('page_history', [])),
                        })

        summary = self._build_summary(runs)
        report = {
            'runs': [asdict(run) for run in runs],
            'summary': summary,
        }
        if self.include_run_details:
            report['run_details'] = run_details

        return report

    def _build_summary(self, runs: Sequence[CrawlerRunResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
        grouped: Dict[str, List[CrawlerRunResult]] = {}
        for run in runs:
            grouped.setdefault(run.crawler_name, []).append(run)

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        metrics = (
            'harvest_rate',
            'precision_at_10',
            'precision_at_20',
            'crawl_time',
            'total_reward',
            'avg_reward',
            'total_crawled',
            'relevant_found',
        )

        for crawler_name, crawler_runs in grouped.items():
            summary[crawler_name] = {}
            for metric in metrics:
                values = [float(getattr(run, metric)) for run in crawler_runs]
                summary[crawler_name][metric] = self._mean_std(values)

        return summary

    @staticmethod
    def save_json(report: Dict[str, object], output_path: Path):
        """Persist a full evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    @staticmethod
    def save_markdown(report: Dict[str, object], output_path: Path):
        """Persist a compact markdown summary table."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '# Crawler Evaluation Summary',
            '',
            '| Crawler | Harvest Rate | P@10 | P@20 | Avg Reward | Crawl Time (s) |',
            '| --- | --- | --- | --- | --- | --- |',
        ]

        summary = report.get('summary', {})
        for crawler_name, metrics in summary.items():
            lines.append(
                '| {name} | {hr:.3f} +/- {hr_std:.3f} | {p10:.3f} | {p20:.3f} | {reward:.2f} | {time:.2f} |'.format(
                    name=crawler_name,
                    hr=metrics['harvest_rate']['mean'],
                    hr_std=metrics['harvest_rate']['std'],
                    p10=metrics['precision_at_10']['mean'],
                    p20=metrics['precision_at_20']['mean'],
                    reward=metrics['avg_reward']['mean'],
                    time=metrics['crawl_time']['mean'],
                )
            )

        output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
