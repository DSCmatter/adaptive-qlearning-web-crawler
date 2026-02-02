"""Quick debug test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph.web_graph import WebGraph

# Test graph operations
graph = WebGraph()
graph.add_page("http://a.com", "<html>A</html>")
graph.add_page("http://b.com", "<html>B</html>")
graph.add_link("http://a.com", "http://b.com", "link to B")

print(f"Nodes: {graph.num_nodes()}")
print(f"Edges: {graph.num_edges()}")
print(f"Neighbors of A: {graph.get_neighbors('http://a.com')}")
