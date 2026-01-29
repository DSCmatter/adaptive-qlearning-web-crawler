"""Evaluation metrics"""

from typing import List


def harvest_rate(relevant_pages: int, total_pages: int) -> float:
    """Harvest Rate: relevant_pages / total_pages"""
    if total_pages == 0:
        return 0.0
    return relevant_pages / total_pages


def target_recall(relevant_found: int, total_relevant: int) -> float:
    """Target Recall: relevant_found / total_relevant_available"""
    if total_relevant == 0:
        return 0.0
    return relevant_found / total_relevant


def precision_at_k(relevant_pages: List[bool], k: int) -> float:
    """Precision in first K pages"""
    if k == 0:
        return 0.0
    return sum(relevant_pages[:k]) / k


def crawl_efficiency(relevant_pages: int, crawl_time: float) -> float:
    """Efficiency: relevant_pages / time_seconds"""
    if crawl_time == 0:
        return 0.0
    return relevant_pages / crawl_time


class CrawlMetrics:
    """Track metrics during crawling"""
    
    def __init__(self):
        self.total_pages = 0
        self.relevant_pages = 0
        self.start_time = None
        self.end_time = None
        self.page_history = []
    
    def add_page(self, is_relevant: bool):
        """Record a crawled page"""
        self.total_pages += 1
        if is_relevant:
            self.relevant_pages += 1
        self.page_history.append(is_relevant)
    
    def get_harvest_rate(self) -> float:
        """Current harvest rate"""
        return harvest_rate(self.relevant_pages, self.total_pages)
    
    def get_precision_at_k(self, k: int) -> float:
        """Precision in first K pages"""
        return precision_at_k(self.page_history, k)
    
    def summary(self) -> dict:
        """Get summary statistics"""
        return {
            'total_pages': self.total_pages,
            'relevant_pages': self.relevant_pages,
            'harvest_rate': self.get_harvest_rate(),
            'precision_at_10': self.get_precision_at_k(10),
            'precision_at_50': self.get_precision_at_k(50),
        }
