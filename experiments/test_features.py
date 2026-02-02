"""Phase 2.3: Test feature extraction on bootstrap graph"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph.web_graph import WebGraph
from models.feature_extractor import FeatureExtractor


def test_feature_extraction():
    print("=" * 60)
    print("Phase 2.3: Test Feature Extraction")
    print("=" * 60)
    
    # Load bootstrap graph
    graph_file = Path(__file__).parent.parent / 'data' / 'graphs' / 'bootstrap_graph.pkl'
    
    if not graph_file.exists():
        print(f"Error: Bootstrap graph not found at {graph_file}")
        print("Run experiments/bootstrap_graph.py first")
        return
    
    print("\nLoading bootstrap graph...")
    graph = WebGraph.load(graph_file)
    print(f"Loaded {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    
    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    extractor = FeatureExtractor()
    
    # Fit TF-IDF on all pages (if we have HTML content)
    # For now, we'll skip this since we don't store HTML in the graph
    # In production, you'd load HTML from storage or database
    
    # Test on a sample of nodes
    nodes = list(graph.graph.nodes())[:10]
    
    print(f"\nTesting feature extraction on {len(nodes)} sample nodes...")
    print("-" * 60)
    
    all_features = []
    
    for i, url in enumerate(nodes):
        print(f"\n[{i+1}/{len(nodes)}] {url[:70]}...")
        
        # Extract individual features
        url_feat = extractor.extract_url_features(url)
        content_feat = extractor.extract_content_features("")  # No HTML stored
        anchor_feat = extractor.extract_anchor_features("test link")
        graph_feat = extractor.extract_graph_features(url, graph)
        
        # Build full context vector
        context = extractor.build_context_vector(
            url=url,
            html="",
            anchor_text="test",
            gnn_embedding=np.zeros(64),
            graph=graph
        )
        
        print(f"  URL features (20): min={url_feat.min():.2f}, max={url_feat.max():.2f}")
        print(f"  Content features (50): min={content_feat.min():.2f}, max={content_feat.max():.2f}")
        print(f"  Anchor features (30): min={anchor_feat.min():.2f}, max={anchor_feat.max():.2f}")
        print(f"  Graph features (10): min={graph_feat.min():.2f}, max={graph_feat.max():.2f}")
        print(f"  Full context (174): shape={context.shape}, has NaN: {np.isnan(context).any()}")
        
        all_features.append(context)
    
    # Summary statistics
    all_features = np.array(all_features)
    
    print("\n" + "=" * 60)
    print("Feature Extraction Summary")
    print("=" * 60)
    print(f"Total features extracted: {all_features.shape[0]}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Min value: {all_features.min():.4f}")
    print(f"Max value: {all_features.max():.4f}")
    print(f"Mean value: {all_features.mean():.4f}")
    print(f"Std value: {all_features.std():.4f}")
    print(f"Has NaN: {np.isnan(all_features).any()}")
    print(f"Has Inf: {np.isinf(all_features).any()}")
    
    # Check dimension consistency
    expected_dim = 174
    if all_features.shape[1] == expected_dim:
        print(f"\n✓ Feature dimension correct: {expected_dim}")
    else:
        print(f"\n✗ Feature dimension mismatch: {all_features.shape[1]} != {expected_dim}")
    
    # Check for valid values
    if not np.isnan(all_features).any() and not np.isinf(all_features).any():
        print("✓ No NaN or Inf values")
    else:
        print("✗ Contains invalid values (NaN or Inf)")
    
    print("\n" + "=" * 60)
    print("✓ Feature extraction test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create labeled training data (Phase 2.2)")
    print("  2. Pre-train GNN on bootstrap graph (Phase 3.1)")
    print("  3. Train hybrid RL agent (Phase 4)")


if __name__ == '__main__':
    test_feature_extraction()
