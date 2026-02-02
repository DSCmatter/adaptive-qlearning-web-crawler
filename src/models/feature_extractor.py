"""Feature extraction from URLs, content, and graph"""

import numpy as np
import re
from urllib.parse import urlparse
from typing import Dict
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class FeatureExtractor:
    """Extract features for bandit context vectors"""
    
    def __init__(self):
        """Initialize feature extractors"""
        # TF-IDF for content features (fit later on corpus)
        self.tfidf = TfidfVectorizer(max_features=40, stop_words='english')
        self.tfidf_fitted = False
        
        # Cache for common computations
        self._domain_cache = {}
    
    def fit_tfidf(self, documents: list):
        """
        Fit TF-IDF vectorizer on document corpus
        
        Args:
            documents: List of text documents (HTML or plain text)
        """
        # Extract text from HTML documents
        texts = []
        for doc in documents:
            try:
                if '<' in doc and '>' in doc:  # Likely HTML
                    soup = BeautifulSoup(doc, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    text = ' '.join(text.split())
                else:
                    text = doc
                texts.append(text)
            except:
                texts.append("")
        
        # Fit vectorizer
        self.tfidf.fit(texts)
        self.tfidf_fitted = True
        print(f"TF-IDF fitted on {len(texts)} documents")
    
    def extract_url_features(self, url: str) -> np.ndarray:
        """
        Extract URL-based features (20 dimensions)
        
        Features:
        - URL structure (depth, length, components)
        - Domain characteristics
        - Special markers (file extensions, parameters)
        """
        parsed = urlparse(url)
        path = parsed.path
        
        # Path analysis
        path_parts = [p for p in path.split('/') if p]
        depth = len(path_parts)
        path_length = len(path)
        
        # Domain analysis
        domain_parts = parsed.netloc.split('.')
        subdomain_count = len(domain_parts) - 2  # Subtract domain + TLD
        
        # Extract features
        features = [
            depth,  # URL depth
            path_length,  # Path length
            len(parsed.netloc),  # Domain length
            subdomain_count,  # Subdomain count
            float(bool(parsed.query)),  # Has query params
            float(bool(parsed.fragment)),  # Has fragment
            float('wiki' in parsed.netloc.lower()),  # Is Wikipedia
            float('.edu' in parsed.netloc),  # Educational domain
            float('.gov' in parsed.netloc),  # Government domain
            float('.org' in parsed.netloc),  # Organization domain
            # File type indicators
            float(path.endswith('.html')),
            float(path.endswith('.pdf')),
            float(path.endswith('.php')),
            float(path.endswith('.asp')),
            # Special path indicators
            float('category' in path.lower()),
            float('archive' in path.lower()),
            float('blog' in path.lower()),
            float('news' in path.lower()),
            len(parsed.query) if parsed.query else 0,  # Query length
            float(any(c.isdigit() for c in path)),  # Contains numbers
        ]
        
        return np.array(features[:20], dtype=np.float32)
    
    def extract_content_features(self, html: str) -> np.ndarray:
        """
        Extract content-based features (50 dimensions)
        
        Features:
        - TF-IDF vector (40 dims - top terms)
        - Content statistics (10 dims)
        """
        if not html:
            return np.zeros(50, dtype=np.float32)
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            text = ' '.join(text.split())  # Normalize whitespace
            
            # TF-IDF features (40 dimensions)
            if self.tfidf_fitted:
                try:
                    tfidf_vec = self.tfidf.transform([text]).toarray()[0]
                except:
                    tfidf_vec = np.zeros(40)
            else:
                tfidf_vec = np.zeros(40)
            
            # Content statistics (10 dimensions)
            word_count = len(text.split())
            char_count = len(text)
            link_count = len(soup.find_all('a'))
            heading_count = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            para_count = len(soup.find_all('p'))
            
            stats = [
                np.log1p(word_count),  # Log-scaled word count
                np.log1p(char_count),  # Log-scaled char count
                np.log1p(link_count),  # Log-scaled link count
                link_count / max(word_count, 1),  # Link density
                heading_count / max(para_count, 1),  # Heading ratio
                float(bool(soup.find('table'))),  # Has tables
                float(bool(soup.find('img'))),  # Has images
                float(bool(soup.find('ul') or soup.find('ol'))),  # Has lists
                len(soup.find_all('div')),  # Div count
                len(soup.find_all('span')),  # Span count
            ]
            
            # Combine TF-IDF and stats
            content_feat = np.concatenate([tfidf_vec, stats])
            return content_feat[:50].astype(np.float32)
            
        except Exception as e:
            # Return zeros if parsing fails
            return np.zeros(50, dtype=np.float32)
    
    def extract_anchor_features(self, anchor_text: str, context: str = "") -> np.ndarray:
        """
        Extract anchor text features (30 dimensions)
        
        Features:
        - Simple bag-of-words features (no external embeddings)
        - Anchor text statistics
        - Keyword presence
        """
        if not anchor_text:
            return np.zeros(30, dtype=np.float32)
        
        anchor_lower = anchor_text.lower()
        words = anchor_lower.split()
        
        # Define keyword categories
        ml_keywords = ['machine', 'learning', 'neural', 'network', 'ai', 'algorithm', 'data', 'model']
        climate_keywords = ['climate', 'environment', 'carbon', 'energy', 'warming', 'sustainability']
        blockchain_keywords = ['blockchain', 'crypto', 'bitcoin', 'ethereum', 'defi', 'smart', 'contract']
        
        features = [
            len(words),  # Word count
            len(anchor_text),  # Character count
            float(any(w.isupper() for w in anchor_text.split())),  # Has capitalized words
            float(any(c.isdigit() for c in anchor_text)),  # Contains numbers
            # ML keywords (5 features)
            sum(kw in anchor_lower for kw in ml_keywords[:5]),
            # Climate keywords (5 features)  
            sum(kw in anchor_lower for kw in climate_keywords[:5]),
            # Blockchain keywords (5 features)
            sum(kw in anchor_lower for kw in blockchain_keywords[:5]),
            # Action words
            float('read' in anchor_lower or 'view' in anchor_lower),
            float('more' in anchor_lower or 'details' in anchor_lower),
            float('article' in anchor_lower or 'page' in anchor_lower),
            # Navigation indicators
            float('main' in anchor_lower or 'home' in anchor_lower),
            float('next' in anchor_lower or 'previous' in anchor_lower),
            float('category' in anchor_lower or 'section' in anchor_lower),
            # Special characters
            float(':' in anchor_text or '|' in anchor_text),
            float('(' in anchor_text or '[' in anchor_text),
        ]
        
        # Pad to 30 dimensions
        while len(features) < 30:
            features.append(0.0)
        
        return np.array(features[:30], dtype=np.float32)
    
    def extract_graph_features(self, node_id: str, graph=None) -> np.ndarray:
        """
        Extract graph structure features (10 dimensions)
        
        Features:
        - In-degree
        - Out-degree  
        - PageRank (if available)
        - Graph position indicators
        """
        if graph is None or not graph.has_node(node_id):
            return np.zeros(10, dtype=np.float32)
        
        try:
            in_deg = graph.get_in_degree(node_id)
            out_deg = graph.get_out_degree(node_id)
            
            # Try to get PageRank (may be cached)
            try:
                pagerank = graph.get_pagerank(node_id)
            except:
                pagerank = 0.0
            
            features = [
                np.log1p(in_deg),  # Log in-degree
                np.log1p(out_deg),  # Log out-degree
                in_deg / max(out_deg, 1),  # In/out ratio
                pagerank * 1000,  # PageRank (scaled)
                float(in_deg > 5),  # Popular page
                float(out_deg > 10),  # Hub page
                float(in_deg == 0),  # Leaf page
                float(out_deg == 0),  # Dead end
                min(in_deg, 10) / 10,  # Normalized in-degree
                min(out_deg, 20) / 20,  # Normalized out-degree
            ]
            
            return np.array(features[:10], dtype=np.float32)
            
        except Exception as e:
            return np.zeros(10, dtype=np.float32)
    
    def build_context_vector(
        self, 
        url: str, 
        html: str = "", 
        anchor_text: str = "",
        gnn_embedding: np.ndarray = None,
        graph=None
    ) -> np.ndarray:
        """
        Build complete context vector (174 dimensions)
        
        Structure:
        - GNN embedding: 64 dim
        - URL features: 20 dim
        - Content features: 50 dim  
        - Anchor features: 30 dim
        - Graph features: 10 dim
        Total: 174 dim
        """
        url_feat = self.extract_url_features(url)
        content_feat = self.extract_content_features(html) if html else np.zeros(50)
        anchor_feat = self.extract_anchor_features(anchor_text)
        graph_feat = self.extract_graph_features(url, graph)
        
        if gnn_embedding is None:
            gnn_embedding = np.zeros(64)
        
        context = np.concatenate([
            gnn_embedding,
            url_feat,
            content_feat,
            anchor_feat,
            graph_feat
        ])
        
        return context
