import time
import numpy as np
import pyvista as pv

from . import Grid
import cupy as cp


def poly2grid(poly, d, x=None, y=None, z=None, l=0):
    """given polydata, find bounds and return meshed grid/plane"""

    xmin, xmax, ymin, ymax, zmin, zmax = poly.bounds

    nx = int((xmax - xmin + 2 * l) / d) + 1
    ny = int((ymax - ymin + 2 * l) / d) + 1
    nz = int((zmax - zmin + 2 * l) / d) + 1

    x = np.linspace(xmin - l, xmax + l, nx) if x is None else float(x)
    y = np.linspace(ymin - l, ymax + l, ny) if y is None else float(y)
    z = np.linspace(zmin - l, zmax + l, nz) if z is None else float(z)

    x, y, z = np.meshgrid(x, y, z)

    grid = pv.StructuredGrid(np.squeeze(x), np.squeeze(y), np.squeeze(z))

    return Grid(grid)


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * cp.linalg.norm(cp.cross(u, v, axis=-1), axis=-1)


def rwg_connectivity(con):
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

    cp = np.stack([internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1)
    cn = np.stack([internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1)

    # define rwg connectivity
    con_m = np.stack([cp, cn])

    # define adjacency
    isin = np.all(np.sort(con) == np.sort(con_m)[:, :, None], axis=-1)

    return con_m, isin


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"Elapsed: {elapsed:.3f} s")
