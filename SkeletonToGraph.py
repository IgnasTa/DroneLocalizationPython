import tifffile
import sknw
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
from scipy.spatial import distance
import tifffile
import sknw
import numpy as np
from skimage import io

def skeleton_to_graph(tiff_path):
    # 1. Load the skeleton image (assumes 0 and 255)
    img = tifffile.imread(tiff_path)
    binary_img = img > 0
    # 2. Build the graph
    # build_sknw takes the skeleton image and returns a NetworkX graph
    graph = sknw.build_sknw(binary_img, multi=False)

    graph = prune_short_branches(graph, 100)
    graph = prune_short_branches(graph, 100)
    graph = prune_short_branches(graph, 50)
    return graph


def prune_short_branches(graph, min_edge_len=50):
    """
    Removes edges that are shorter than min_edge_len and the
    resulting dangling leaf nodes.
    """
    nodes_to_remove = []

    # Iterate through edges
    for (s, e) in list(graph.edges()):
        # Get the edge length (number of pixels in the segment)
        edge_len = len(graph[s][e]['pts'])

        # If the edge is short
        if edge_len < min_edge_len:
            # Check if one of the endpoints is a leaf
            if graph.degree(s) == 1:
                nodes_to_remove.append(s)
            if graph.degree(e) == 1:
                nodes_to_remove.append(e)

    # Remove the nodes
    graph.remove_nodes_from(nodes_to_remove)
    print(f"Pruned {len(nodes_to_remove)} nodes with short branches.")
    return graph


