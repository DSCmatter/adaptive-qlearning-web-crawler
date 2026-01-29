"""Tests for crawler implementations"""

import pytest
from src.crawler.base_crawler import BaseCrawler


def test_base_crawler_init():
    """Test BaseCrawler initialization"""
    crawler = BaseCrawler(max_pages=10)
    assert crawler.max_pages == 10
    assert len(crawler.visited) == 0


def test_url_validation():
    """Test URL filtering"""
    crawler = BaseCrawler()
    
    # Valid URLs
    assert crawler.is_valid_url('https://example.com/page')
    
    # Invalid URLs
    assert not crawler.is_valid_url('https://example.com/page#section')  # Fragment
    assert not crawler.is_valid_url('https://example.com/file.pdf')  # PDF
    assert not crawler.is_valid_url('ftp://example.com')  # Wrong scheme


# TODO: Add more tests for other components
