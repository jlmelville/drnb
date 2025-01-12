import scipy.sparse

from drnb.neighbors.hubness import nn_to_sparse


def umap_graph_binary(knn, edge_weight: float = 1.0) -> scipy.sparse.coo_matrix:
    """Create a symmetrized binary adjacency matrix from a nearest neighbor graph."""
    graph = nn_to_sparse(knn)
    graph = umap_symmetrize(graph)
    graph.data.fill(edge_weight)
    return graph


def umap_symmetrize(
    graph: scipy.sparse.coo_matrix, set_op_mix_ratio: float = 1.0
) -> scipy.sparse.coo_matrix:
    """Symmetrize a weighted adjacency matrix in the UMAP style."""
    transpose = graph.transpose()

    prod_matrix = graph.multiply(transpose)

    graph = (
        set_op_mix_ratio * (graph + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )

    graph.eliminate_zeros()
    return graph
