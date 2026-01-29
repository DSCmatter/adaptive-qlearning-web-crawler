"""Quick test script to validate project skeleton"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from crawler.base_crawler import BaseCrawler
        print("  ✓ BaseCrawler")
        
        from crawler.adaptive_crawler import AdaptiveCrawler
        print("  ✓ AdaptiveCrawler")
        
        from models.gnn_encoder import WebGraphEncoder
        print("  ✓ WebGraphEncoder")
        
        from models.contextual_bandit import LinUCBBandit
        print("  ✓ LinUCBBandit")
        
        from models.qlearning_agent import QLearningAgent
        print("  ✓ QLearningAgent")
        
        from models.feature_extractor import FeatureExtractor
        print("  ✓ FeatureExtractor")
        
        from environment.reward_function import RewardFunction
        print("  ✓ RewardFunction")
        
        from graph.web_graph import WebGraph
        print("  ✓ WebGraph")
        
        from graph.graph_builder import bootstrap_initial_graph
        print("  ✓ bootstrap_initial_graph")
        
        from utils.url_utils import normalize_url, filter_candidate_links
        print("  ✓ url_utils")
        
        from utils.metrics import CrawlMetrics
        print("  ✓ CrawlMetrics")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_basic_functionality():
    """Test basic functionality of core components"""
    print("Testing basic functionality...")
    
    try:
        # Test BaseCrawler
        from crawler.base_crawler import BaseCrawler
        crawler = BaseCrawler(max_pages=10)
        assert crawler.max_pages == 10
        assert crawler.is_valid_url('https://example.com')
        assert not crawler.is_valid_url('https://example.com/file.pdf')
        print("  ✓ BaseCrawler works")
        
        # Test WebGraph
        from graph.web_graph import WebGraph
        graph = WebGraph()
        graph.add_page('https://example.com', html='<html></html>')
        graph.add_page('https://example.com/page2')
        graph.add_link('https://example.com', 'https://example.com/page2')
        assert graph.num_nodes() == 2
        assert graph.num_edges() == 1
        print("  ✓ WebGraph works")
        
        # Test RewardFunction
        from environment.reward_function import RewardFunction
        reward_fn = RewardFunction()
        reward = reward_fn.compute_reward(
            is_relevant=True,
            relevance_score=0.9,
            depth=2
        )
        assert reward > 0  # Should be positive for relevant page
        print("  ✓ RewardFunction works")
        
        # Test LinUCBBandit
        from models.contextual_bandit import LinUCBBandit
        import numpy as np
        bandit = LinUCBBandit(context_dim=10)
        links = ['url1', 'url2', 'url3']
        contexts = [np.random.rand(10) for _ in range(3)]
        selected, idx = bandit.select_link(links, contexts)
        assert selected in links
        print("  ✓ LinUCBBandit works")
        
        # Test QLearningAgent
        from models.qlearning_agent import QLearningAgent
        agent = QLearningAgent(state_dim=10, action_dim=2)
        state = np.random.rand(10)
        action = agent.get_action(state, [0, 1])
        assert action in [0, 1]
        print("  ✓ QLearningAgent works")
        
        # Test FeatureExtractor
        from models.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        url_features = extractor.extract_url_features('https://example.com/page')
        assert len(url_features) == 20
        print("  ✓ FeatureExtractor works")
        
        # Test CrawlMetrics
        from utils.metrics import CrawlMetrics
        metrics = CrawlMetrics()
        metrics.add_page(is_relevant=True)
        metrics.add_page(is_relevant=False)
        assert metrics.total_pages == 2
        assert metrics.relevant_pages == 1
        assert metrics.get_harvest_rate() == 0.5
        print("  ✓ CrawlMetrics works")
        
        # Test URL utilities
        from utils.url_utils import normalize_url, get_domain
        url = normalize_url('https://example.com/page#section')
        assert '#' not in url
        domain = get_domain('https://example.com/page')
        assert domain == 'example.com'
        print("  ✓ URL utilities work")
        
        print("\n✅ All basic functionality tests passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_encoder():
    """Test GNN encoder (requires torch-geometric)"""
    print("Testing GNN encoder...")
    
    try:
        import torch
        from models.gnn_encoder import WebGraphEncoder
        
        # Create small test graph
        encoder = WebGraphEncoder(input_dim=10, hidden_dim=16, output_dim=8)
        
        # Mock data
        x = torch.randn(5, 10)  # 5 nodes, 10 features each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Simple chain
        
        # Forward pass
        embeddings = encoder(x, edge_index)
        assert embeddings.shape == (5, 8)
        
        print("  ✓ GNN encoder works")
        print(f"    Input: {x.shape}, Output: {embeddings.shape}")
        
        # Test freeze
        encoder.freeze()
        for param in encoder.parameters():
            assert not param.requires_grad
        print("  ✓ GNN freeze works")
        
        print("\n✅ GNN encoder test passed!\n")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping GNN test (missing dependency): {e}\n")
        return True  # Not a failure, just skip
    except Exception as e:
        print(f"\n❌ GNN test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("TESTING PROJECT SKELETON")
    print("="*60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test GNN encoder
    results.append(("GNN Encoder", test_gnn_encoder()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print()
    
    if all(passed for _, passed in results):
        print("🎉 All tests passed! Project skeleton is working.")
        print()
        print("Next steps (WALKTHROUGH.md Week 2):")
        print("  1. Collect seed URLs for target domain")
        print("  2. Create labeled training data")
        print("  3. Bootstrap initial graph (500 pages)")
        return 0
    else:
        print("⚠ Some tests failed. Check errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
