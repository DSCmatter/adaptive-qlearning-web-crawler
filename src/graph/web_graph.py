"""Web graph data structure"""

import networkx as nx
from typing import Dict, List, Optional


class WebGraph:
    """
    Manages web graph structure
    Stores nodes (pages) and edges (links)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_features: Dict[str, dict] = {}
        self.node_labels: Dict[str, int] = {}  # For training
    
    def add_page(self, url: str, html: str = "", features: dict = None, label: Optional[int] = None):
        """
        Add page to graph
        
        Args:
            url: Page URL
            html: Page HTML content
            features: Feature dictionary
            label: Relevance label (0/1) for training
        """
        self.graph.add_node(url)
        
        if features:
            self.node_features[url] = features
        
        if label is not None:
            self.node_labels[url] = label
    
    def add_link(self, from_url: str, to_url: str, anchor_text: str = ""):
        """
        Add directed edge (link) between pages
        
        Args:
            from_url: Source page
            to_url: Destination page
            anchor_text: Link anchor text
        """
        self.graph.add_edge(from_url, to_url, anchor_text=anchor_text)
    
    def get_neighbors(self, url: str) -> List[str]:
        """Get outgoing links from a page"""
        return list(self.graph.successors(url))
    
    def get_in_degree(self, url: str) -> int:
        """Get number of incoming links"""
        return self.graph.in_degree(url)
    
    def get_out_degree(self, url: str) -> int:
        """Get number of outgoing links"""
        return self.graph.out_degree(url)
    
    def get_pagerank(self, url: str) -> float:
        """Get PageRank score (compute if not cached)"""
        if not hasattr(self, '_pagerank'):
            self._pagerank = nx.pagerank(self.graph)
        return self._pagerank.get(url, 0.0)
    
    def num_nodes(self) -> int:
        """Total number of pages"""
        return self.graph.number_of_nodes()
    
    def num_edges(self) -> int:
        """Total number of links"""
        return self.graph.number_of_edges()
    
    def has_node(self, url: str) -> bool:
        """Check if page exists in graph"""
        return url in self.graph
    
    def get_label(self, url: str) -> Optional[int]:
        """Get relevance label for training"""
        return self.node_labels.get(url)
