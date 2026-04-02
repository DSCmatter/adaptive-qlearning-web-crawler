"""
Phase 6: Evaluation and baseline comparison runner.

This script compares the hybrid crawler against heuristic and ablation
baselines, then writes JSON and markdown summaries to ``data/results``.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crawler.adaptive_crawler import AdaptiveCrawler
from crawler.baseline_crawlers import BestFirstCrawler, PageRankCrawler, RandomCrawler
from evaluation import CrawlerEvaluator, CrawlerSpec
from models.contextual_bandit import LinUCBBandit
from models.feature_extractor import FeatureExtractor
from models.gnn_encoder import WebGraphEncoder
from models.qlearning_agent import QLearningAgent
from utils.relevance import TopicDefinition, TopicRelevanceScorer


def parse_args():
    parser = argparse.ArgumentParser(description='Run Phase 6 crawler evaluation.')
    parser.add_argument('--max-pages', type=int, default=20, help='Per-run crawl budget.')
    parser.add_argument('--runs-per-seed', type=int, default=1, help='How many repeated runs per seed.')
    parser.add_argument('--max-seeds-per-topic', type=int, default=2, help='Number of seeds to use from each topic file.')
    parser.add_argument('--output-prefix', default='PHASE_6_EVAL', help='Prefix for output files.')
    return parser.parse_args()


def load_config(base_dir: Path):
    config_path = base_dir / 'configs' / 'crawler_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def load_seed_configs(base_dir: Path, max_per_topic: int):
    seed_dir = base_dir / 'data' / 'seeds'
    seed_urls: List[str] = []
    topic_by_seed: Dict[str, TopicDefinition] = {}
    topic_names: List[str] = []

    for seed_file in sorted(seed_dir.glob('*_seeds.json')):
        with open(seed_file, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        topic_names.append(payload.get('topic', seed_file.stem))
        current_keywords = payload.get('keywords', [])
        negative_keywords = []
        for other_seed_file in sorted(seed_dir.glob('*_seeds.json')):
            if other_seed_file == seed_file:
                continue
            with open(other_seed_file, 'r', encoding='utf-8') as handle:
                other_payload = json.load(handle)
            negative_keywords.extend(other_payload.get('keywords', []))

        topic_definition = TopicDefinition(
            name=payload.get('topic', seed_file.stem),
            keywords=current_keywords,
            negative_keywords=negative_keywords,
        )

        for seed_url in payload.get('seeds', [])[:max_per_topic]:
            seed_urls.append(seed_url)
            topic_by_seed[seed_url] = topic_definition

    return seed_urls, topic_by_seed, topic_names


def build_feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()


def build_gnn(config, models_dir: Path):
    weights_path = models_dir / 'gnn_encoder_frozen.pt'
    if not weights_path.exists():
        return None

    gnn_config = config.get('gnn', {})
    encoder = WebGraphEncoder(
        input_dim=gnn_config.get('input_dim', 174),
        hidden_dim=gnn_config.get('hidden_dim', 128),
        output_dim=gnn_config.get('output_dim', 64),
        num_layers=gnn_config.get('num_layers', 2),
        dropout=gnn_config.get('dropout', 0.3),
    )
    encoder.load_state_dict(torch.load(weights_path, weights_only=True))
    encoder.eval()
    return encoder


def build_bandit(config, models_dir: Path):
    bandit_config = config.get('bandit', {})
    bandit = LinUCBBandit(
        context_dim=bandit_config.get('context_dim', 174),
        alpha=bandit_config.get('alpha', 1.0),
    )
    weights_path = models_dir / 'bandit_arms.pkl'
    if weights_path.exists():
        with open(weights_path, 'rb') as handle:
            bandit.arms = pickle.load(handle)
    return bandit


def build_q_agent(config, models_dir: Path):
    weights_path = models_dir / 'qlearning_agent.pt'
    if not weights_path.exists():
        return None

    q_config = config.get('qlearning', {})
    agent = QLearningAgent(
        state_dim=q_config.get('state_dim', 69),
        action_dim=q_config.get('action_dim', 2),
        hidden_dim=q_config.get('hidden_dim', 64),
        learning_rate=q_config.get('learning_rate', 0.001),
        gamma=q_config.get('gamma', 0.95),
        epsilon=q_config.get('epsilon', 0.1),
    )
    agent.load_state_dict(torch.load(weights_path, weights_only=True))
    agent.epsilon = 0.01
    return agent


def build_relevance_fn(seed_url: str, topic_by_seed: Dict[str, TopicDefinition]):
    topic_definition = topic_by_seed[seed_url]
    scorer = TopicRelevanceScorer(topic_definition)
    return scorer.is_relevant


def build_hybrid_factory(
    config,
    models_dir: Path,
    crawler_config,
    max_pages: int,
    mode: str,
    topic_by_seed: Dict[str, TopicDefinition]
):
    def factory(seed_url: str):
        feature_extractor = build_feature_extractor()
        q_agent = build_q_agent(config, models_dir) if mode in {'hybrid', 'pure_q', 'no_gnn'} else None
        bandit = build_bandit(config, models_dir) if mode in {'hybrid', 'pure_bandit', 'no_gnn'} else None
        gnn_encoder = build_gnn(config, models_dir) if mode == 'hybrid' else None

        return AdaptiveCrawler(
            gnn_encoder=gnn_encoder,
            qlearning_agent=q_agent,
            bandit=bandit,
            feature_extractor=feature_extractor,
            max_pages=max_pages,
            delay=crawler_config.get('delay', 1.0),
            timeout=crawler_config.get('timeout', 10),
            max_candidates=crawler_config.get('max_candidates', 50),
            relevance_fn=build_relevance_fn(seed_url, topic_by_seed),
        )

    return factory


def build_specs(config, models_dir: Path, max_pages: int, topic_by_seed: Dict[str, TopicDefinition]):
    crawler_config = config.get('crawler', {})
    specs = [
        CrawlerSpec(
            name='random',
            factory=lambda seed_url: RandomCrawler(
                max_pages=max_pages,
                delay=crawler_config.get('delay', 1.0),
                timeout=crawler_config.get('timeout', 10),
                max_candidates=crawler_config.get('max_candidates', 50),
                relevance_fn=build_relevance_fn(seed_url, topic_by_seed),
            ),
        ),
        CrawlerSpec(
            name='best_first',
            factory=lambda seed_url: BestFirstCrawler(
                max_pages=max_pages,
                delay=crawler_config.get('delay', 1.0),
                timeout=crawler_config.get('timeout', 10),
                max_candidates=crawler_config.get('max_candidates', 50),
                relevance_fn=build_relevance_fn(seed_url, topic_by_seed),
            ),
        ),
        CrawlerSpec(
            name='pagerank',
            factory=lambda seed_url: PageRankCrawler(
                max_pages=max_pages,
                delay=crawler_config.get('delay', 1.0),
                timeout=crawler_config.get('timeout', 10),
                max_candidates=crawler_config.get('max_candidates', 50),
                relevance_fn=build_relevance_fn(seed_url, topic_by_seed),
            ),
        ),
    ]

    if (models_dir / 'qlearning_agent.pt').exists():
        specs.append(CrawlerSpec(
            name='pure_q',
            factory=build_hybrid_factory(config, models_dir, crawler_config, max_pages, mode='pure_q', topic_by_seed=topic_by_seed),
        ))

    if (models_dir / 'bandit_arms.pkl').exists():
        specs.append(CrawlerSpec(
            name='pure_bandit',
            factory=build_hybrid_factory(config, models_dir, crawler_config, max_pages, mode='pure_bandit', topic_by_seed=topic_by_seed),
        ))

    if (models_dir / 'qlearning_agent.pt').exists() and (models_dir / 'bandit_arms.pkl').exists():
        specs.append(CrawlerSpec(
            name='hybrid_no_gnn',
            factory=build_hybrid_factory(config, models_dir, crawler_config, max_pages, mode='no_gnn', topic_by_seed=topic_by_seed),
        ))

    if (
        (models_dir / 'gnn_encoder_frozen.pt').exists()
        and (models_dir / 'qlearning_agent.pt').exists()
        and (models_dir / 'bandit_arms.pkl').exists()
    ):
        specs.append(CrawlerSpec(
            name='hybrid',
            factory=build_hybrid_factory(config, models_dir, crawler_config, max_pages, mode='hybrid', topic_by_seed=topic_by_seed),
        ))

    return specs


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'data' / 'models'
    results_dir = base_dir / 'data' / 'results'

    config = load_config(base_dir)
    seed_urls, topic_by_seed, topic_names = load_seed_configs(base_dir, max_per_topic=args.max_seeds_per_topic)
    specs = build_specs(config, models_dir, max_pages=args.max_pages, topic_by_seed=topic_by_seed)

    if not seed_urls:
        raise RuntimeError('No evaluation seeds were found in data/seeds.')
    if not specs:
        raise RuntimeError('No crawler configurations are available for evaluation.')

    evaluator = CrawlerEvaluator(
        crawler_specs=specs,
        seed_urls=seed_urls,
        runs_per_seed=args.runs_per_seed,
    )
    report = evaluator.evaluate()

    json_path = results_dir / f'{args.output_prefix}.json'
    markdown_path = results_dir / f'{args.output_prefix}.md'
    CrawlerEvaluator.save_json(report, json_path)
    CrawlerEvaluator.save_markdown(report, markdown_path)

    print('=' * 72)
    print('Phase 6 Evaluation Summary')
    print('=' * 72)
    print(f'Seeds evaluated: {len(seed_urls)}')
    print(f'Topics:          {", ".join(topic_names)}')
    print(f'Crawler configs: {", ".join(spec.name for spec in specs)}')
    print(f'JSON report:     {json_path}')
    print(f'Markdown report: {markdown_path}')
    print('')

    for crawler_name, metrics in report['summary'].items():
        print(
            f"{crawler_name:12s} "
            f"HR={metrics['harvest_rate']['mean']:.3f} "
            f"P@10={metrics['precision_at_10']['mean']:.3f} "
            f"Reward={metrics['avg_reward']['mean']:.2f}"
        )


if __name__ == '__main__':
    main()
