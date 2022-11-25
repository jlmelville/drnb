from drnb.neighbors.hubness import nn_to_sparse


def umap_graph_binary(knn, edge_weight=1.0):
    graph = nn_to_sparse(knn)
    graph = umap_symmetrize(graph)
    graph.data.fill(edge_weight)
    return graph


def umap_symmetrize(graph, set_op_mix_ratio=1.0):
    transpose = graph.transpose()

    prod_matrix = graph.multiply(transpose)

    graph = (
        set_op_mix_ratio * (graph + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )

    graph.eliminate_zeros()
    return graph
