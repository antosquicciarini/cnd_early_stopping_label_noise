import networkx as nx
import numpy as np
from network_graph import graph_DNN_class_based, graph_DNN_class_based_selected_layers
# Compute Von Neumann Entropy of the extracted graph
def von_neumann_entropy(G):
    # Compute Laplacian Matrix
    undirected_G = G.to_undirected()

    L = nx.laplacian_matrix(undirected_G).todense()

    # Calculate the trace of L
    trace_L = np.trace(L)

    # Normalize the Laplacian matrix
    if trace_L != 0:
        normalized_L = L / trace_L
    else:
        raise ValueError("Trace of the Laplacian matrix is zero, normalization cannot be performed.")

    eigvals = np.linalg.eigvals(normalized_L)
    eigvals = np.maximum(eigvals, 1e-12)  # Avoid log of zero issues
    entropy = -np.sum(eigvals * np.log(eigvals))
    return entropy

# Compute Shannon Entropy of the graph
def shannon_entropy(G):
    # Compute edge weights
    weights = np.array([data['weight'] for u, v, data in G.edges(data=True)])
    if len(weights) == 0:
        return 0.0

    # Compute probability distribution
    weight_sum = np.sum(weights)
    probabilities = weights / weight_sum

    # Compute Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))  # Adding epsilon to avoid log(0)
    return entropy


def graph_entropy_evaluation(loader, model, device):

    selected_layers = [1, 2, 3]

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        graph_list = graph_DNN_class_based_selected_layers(model, images, labels, selected_layers)
        break

    CGE = []
    for G in graph_list:
        CGE.append(von_neumann_entropy(G))
    return CGE
