"""Topic-aware relevance scoring utilities for evaluation."""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup


WIKIPEDIA_NON_CONTENT_NAMESPACES = (
    'special:',
    'wikipedia:',
    'help:',
    'portal:',
    'talk:',
    'category:',
    'template:',
    'template_talk:',
    'file:',
    'user:',
    'mediawiki:',
)


@dataclass(frozen=True)
class TopicDefinition:
    """A crawl topic and its evaluation vocabulary."""

    name: str
    keywords: Sequence[str]
    negative_keywords: Sequence[str]


def _normalize_text(value: str) -> str:
    return re.sub(r'\s+', ' ', value.lower()).strip()


def _keyword_tokens(keywords: Iterable[str]) -> List[str]:
    tokens = set()
    for keyword in keywords:
        for token in re.findall(r'[a-z0-9]+', keyword.lower()):
            if len(token) >= 4:
                tokens.add(token)
    return sorted(tokens)


class TopicRelevanceScorer:
    """Heuristic scorer that is stricter and topic-specific."""

    def __init__(self, topic: TopicDefinition):
        self.topic = topic
        self.topic_keywords = [_normalize_text(keyword) for keyword in topic.keywords]
        self.negative_keywords = [_normalize_text(keyword) for keyword in topic.negative_keywords]
        self.topic_tokens = _keyword_tokens(self.topic_keywords)

    def _is_content_url(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if 'wikipedia.org' in host and not host.startswith('en.'):
            return False

        if any(namespace in path for namespace in WIKIPEDIA_NON_CONTENT_NAMESPACES):
            return False

        if any(blocked in host for blocked in ('donate.wikimedia.org', 'auth.wikimedia.org', 'web.archive.org')):
            return False

        return True

    def _extract_text(self, url: str, html: str) -> str:
        if not html:
            return _normalize_text(url)

        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        title = soup.title.get_text(' ', strip=True) if soup.title else ''
        headings = ' '.join(
            heading.get_text(' ', strip=True)
            for heading in soup.find_all(['h1', 'h2'], limit=4)
        )
        body_text = soup.get_text(' ', strip=True)
        return _normalize_text(f'{url} {title} {headings} {body_text[:4000]}')

    def score(self, url: str, html: str) -> Dict[str, int]:
        text = self._extract_text(url, html)
        topic_phrase_hits = sum(keyword in text for keyword in self.topic_keywords)
        token_hits = sum(token in text for token in self.topic_tokens)
        negative_hits = sum(keyword in text for keyword in self.negative_keywords)
        title_and_url = _normalize_text(url)

        strong_phrase_hits = sum(keyword in title_and_url for keyword in self.topic_keywords)
        return {
            'topic_phrase_hits': topic_phrase_hits,
            'token_hits': token_hits,
            'negative_hits': negative_hits,
            'strong_phrase_hits': strong_phrase_hits,
        }

    def is_relevant(self, url: str, html: str) -> bool:
        if not self._is_content_url(url):
            return False

        hits = self.score(url, html)

        if hits['negative_hits'] > hits['topic_phrase_hits']:
            return False

        if hits['strong_phrase_hits'] >= 1 and hits['topic_phrase_hits'] >= 1:
            return True

        if hits['topic_phrase_hits'] >= 2:
            return True

        if hits['topic_phrase_hits'] >= 1 and hits['token_hits'] >= 4:
            return True

        return False
