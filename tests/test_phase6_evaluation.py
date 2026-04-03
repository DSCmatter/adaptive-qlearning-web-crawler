"""Phase 6 evaluation tests using an in-memory mini web."""

from src.crawler.adaptive_crawler import AdaptiveCrawler
from src.crawler.baseline_crawlers import BestFirstCrawler, RandomCrawler
from src.evaluation.crawler_evaluator import CrawlerEvaluator, CrawlerSpec
from src.models.feature_extractor import FeatureExtractor
from src.utils.relevance import TopicDefinition, TopicRelevanceScorer


SEED_URL = 'https://example.com/start'
ML_URL = 'https://example.com/machine-learning'
COOKING_URL = 'https://example.com/cooking'
ADVANCED_ML_URL = 'https://example.com/neural-networks'

FAKE_WEB = {
    SEED_URL: f'''
        <html><body>
            <p>General index page.</p>
            <a href="{COOKING_URL}">Cooking tips</a>
            <a href="{ML_URL}">Read more</a>
        </body></html>
    ''',
    ML_URL: f'''
        <html><body>
            <h1>Machine learning</h1>
            <p>Deep learning and neural network tutorial.</p>
            <a href="{ADVANCED_ML_URL}">Neural network notes</a>
        </body></html>
    ''',
    COOKING_URL: '''
        <html><body>
            <h1>Cooking</h1>
            <p>Soup recipes and kitchen basics.</p>
        </body></html>
    ''',
    ADVANCED_ML_URL: '''
        <html><body>
            <h1>Neural networks</h1>
            <p>Artificial intelligence and deep learning content.</p>
        </body></html>
    ''',
}


def fake_fetch(url: str):
    return FAKE_WEB.get(url)


def relevance_fn(url: str, html: str) -> bool:
    text = html.lower()
    return 'machine learning' in text or 'neural network' in text or 'deep learning' in text


def test_topic_relevance_scorer_rejects_non_content_and_wrong_topic():
    scorer = TopicRelevanceScorer(
        TopicDefinition(
            name='machine_learning',
            keywords=['machine learning', 'deep learning', 'neural network'],
            negative_keywords=['climate change', 'renewable energy'],
        )
    )

    assert scorer.is_relevant(
        'https://en.wikipedia.org/wiki/Machine_learning',
        '<html><head><title>Machine learning</title></head><body>Deep learning and neural network methods.</body></html>',
    )
    assert not scorer.is_relevant(
        'https://en.wikipedia.org/wiki/Climate_change',
        '<html><head><title>Climate change</title></head><body>Renewable energy and global warming discussion.</body></html>',
    )
    assert not scorer.is_relevant(
        'https://en.wikipedia.org/wiki/Special:Random',
        '<html><body>Machine learning</body></html>',
    )


def test_best_first_crawler_prioritizes_relevant_candidate():
    crawler = BestFirstCrawler(
        max_pages=2,
        fetch_callback=fake_fetch,
        relevance_fn=relevance_fn,
    )

    result = crawler.crawl(SEED_URL)

    assert result['total_crawled'] == 2
    assert result['relevant_found'] == 1
    assert result['page_history'] == [False, True]
    assert result['precision_at_10'] == 0.5


def test_adaptive_crawler_returns_phase6_metrics_without_models():
    crawler = AdaptiveCrawler(
        max_pages=2,
        fetch_callback=fake_fetch,
        relevance_fn=relevance_fn,
    )

    result = crawler.crawl(SEED_URL)

    assert result['total_crawled'] == 2
    assert 'precision_at_10' in result
    assert 'crawl_time' in result
    assert len(result['trace']) == 2


def test_adaptive_candidate_context_uses_anchor_and_source_page_context():
    crawler = AdaptiveCrawler(feature_extractor=FeatureExtractor())

    context = crawler._build_candidate_context(
        ML_URL,
        {
            ML_URL: {
                'anchor_text': 'Machine learning guide',
                'source_html': '<html><body>Deep learning and neural network overview.</body></html>',
            }
        },
    )

    content_slice = context[84:134]
    anchor_slice = context[134:164]

    assert content_slice.sum() > 0
    assert anchor_slice.sum() > 0


def test_crawler_evaluator_aggregates_summary_metrics():
    evaluator = CrawlerEvaluator(
        crawler_specs=[
            CrawlerSpec(
                name='best_first',
                factory=lambda: BestFirstCrawler(
                    max_pages=2,
                    fetch_callback=fake_fetch,
                    relevance_fn=relevance_fn,
                ),
            ),
            CrawlerSpec(
                name='random',
                factory=lambda: RandomCrawler(
                    max_pages=2,
                    fetch_callback=fake_fetch,
                    relevance_fn=relevance_fn,
                    random_seed=7,
                ),
            ),
        ],
        seed_urls=[SEED_URL],
        runs_per_seed=1,
    )

    report = evaluator.evaluate()

    assert len(report['runs']) == 2
    assert set(report['summary']) == {'best_first', 'random'}
    assert report['summary']['best_first']['harvest_rate']['mean'] == 0.5
    assert report['summary']['best_first']['precision_at_10']['mean'] == 0.5


def test_crawler_evaluator_can_include_run_details():
    evaluator = CrawlerEvaluator(
        crawler_specs=[
            CrawlerSpec(
                name='best_first',
                factory=lambda: BestFirstCrawler(
                    max_pages=2,
                    fetch_callback=fake_fetch,
                    relevance_fn=relevance_fn,
                ),
            ),
        ],
        seed_urls=[SEED_URL],
        runs_per_seed=1,
        include_run_details=True,
    )

    report = evaluator.evaluate()

    assert 'run_details' in report
    assert len(report['run_details']) == 1
    detail = report['run_details'][0]
    assert detail['crawler_name'] == 'best_first'
    assert detail['seed_url'] == SEED_URL
    assert isinstance(detail['trace'], list)


def test_adaptive_crawler_emits_diagnostics_when_enabled():
    crawler = AdaptiveCrawler(
        max_pages=2,
        fetch_callback=fake_fetch,
        relevance_fn=relevance_fn,
        enable_diagnostics=True,
        online_updates=False,
    )

    result = crawler.crawl(SEED_URL)

    assert 'diagnostics' in result
    assert len(result['diagnostics']) == 2
    assert result['trace'][0]['selection_mode'] == 'frontier_order'
    assert result['trace'][0]['candidate_count'] == 1
