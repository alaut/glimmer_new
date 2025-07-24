import numpy as np


def setup_rwg_connectivity(con):
    """generate RWG (Rao, Wilton, Glisson) connectivity from trimesh face connectivity"""

    # construct edges from face connectivity (face x edge x point) (N, 3, 2)
    edges = con[:, [[0, 1], [1, 2], [2, 0]]]

    # sort flattened edges by points (N x 3, 2)
    sorted_edges = np.sort(edges.reshape(-1, 2))

    # count edge uniqueness
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)

    # define rwg basis of unique internal edges (M, 2)
    internal_edges = unique_edges[counts == 2]

    def get_free_vert(edge):
        """find free vertex given edge (2,) in face edges (M, 3, 2)"""

        # match edge to face
        face = np.any(np.all(edge == edges, axis=-1), axis=-1)

        # get face to points
        points = np.unique(edges[face])

        # find free vertex
        vertex = np.setdiff1d(points, edge)

        return int(vertex[0])

    # find free rwg vertices
    vert_pos = np.array([get_free_vert(edge) for edge in internal_edges])
    vert_neg = np.array([get_free_vert(edge[::-1]) for edge in internal_edges])

    con_m = np.stack(
        [
            np.stack([internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1),
            np.stack([internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1),
        ]
    )
    return con_m
