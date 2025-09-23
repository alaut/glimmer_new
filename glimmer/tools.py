from pyacvd import Clustering
from dataclasses import dataclass
import time
import numpy as np
import pyvista as pv


import cupy as cp


def poly2grid(
    ds: pv.DataSet,
    d: float,
    x: float = None,
    y: float = None,
    z: float = None,
    l: float = 0,
) -> pv.StructuredGrid:
    """given pyvista DataSet, find bounds and return meshed grid/plane"""

    xmin, xmax, ymin, ymax, zmin, zmax = ds.bounds

    nx = int((xmax - xmin + 2 * l) / d) + 1
    ny = int((ymax - ymin + 2 * l) / d) + 1
    nz = int((zmax - zmin + 2 * l) / d) + 1

    x = np.linspace(xmin - l, xmax + l, nx) if x is None else float(x)
    y = np.linspace(ymin - l, ymax + l, ny) if y is None else float(y)
    z = np.linspace(zmin - l, zmax + l, nz) if z is None else float(z)

    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    grid = pv.StructuredGrid(np.squeeze(x), np.squeeze(y), np.squeeze(z))
    grid._name = f"Volume {ds.bounds}"

    return grid


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * cp.linalg.norm(cp.cross(u, v, axis=-1), axis=-1)


# def rwg_connectivity(con):
#     """generate RWG (Rao, Wilton, Glisson) connectivity from trimesh face connectivity"""

#     # construct edges from face connectivity (face x edge x point) (N, 3, 2)
#     edges = con[:, [[0, 1], [1, 2], [2, 0]]]

#     # sort flattened edges by points (N x 3, 2)
#     sorted_edges = np.sort(edges.reshape(-1, 2))

#     # count edge uniqueness
#     unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)

#     # define rwg basis of unique internal edges (M, 2)
#     internal_edges = unique_edges[counts == 2]

#     def get_free_vert(edge):
#         """find free vertex given edge (2,) in face edges (M, 3, 2)"""

#         # match edge to face
#         face = np.any(np.all(edge == edges, axis=-1), axis=-1)

#         # get face to points
#         points = np.unique(edges[face])

#         # find free vertex
#         vertex = np.setdiff1d(points, edge)

#         return int(vertex[0])

#     # find free rwg vertices
#     vert_pos = np.array([get_free_vert(edge) for edge in internal_edges])
#     vert_neg = np.array([get_free_vert(edge[::-1]) for edge in internal_edges])

#     cp = np.stack([internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1)
#     cn = np.stack([internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1)

#     # define rwg connectivity
#     con_m = np.stack([cp, cn])

#     # define adjacency
#     isin = np.all(np.sort(con) == np.sort(con_m)[:, :, None], axis=-1)

#     return con_m, isin
#


def remesh(poly: pv.PolyData, dl=None, subdivisions=None, target_vertices=None):
    """remesh a surface mesh using clustering algorithm"""

    # Perform clustering with pyacvd
    clus = Clustering(poly.copy())

    if dl is not None:
        target_area = 3**0.5 / 4 * dl**2
        target_faces = int(np.ceil(poly.area / target_area))
        print(f"target_faces: {target_faces}")

    if target_vertices is None:
        target_vertices = int(target_faces / 2)
        print(f"target_vertices: {target_vertices}")

    if subdivisions is None:
        subdivisions = max(1, int(np.ceil(np.log2(target_faces / poly.n_cells))))
        print(f"subdivisions: {subdivisions}")

    clus.subdivide(subdivisions)
    clus.cluster(target_vertices)

    # Create remeshed output
    remeshed = clus.create_mesh()

    remeshed = remeshed.interpolate(poly, radius=1e-3)

    return remeshed


@dataclass
class Timer:

    text: str = "Elapsed"

    def __enter__(self):
        self.start = time.time()
        print(self.text, end=" ")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[{elapsed:.1f} s]")


def integrate_power(ds: pv.DataSet):

    grid = ds.extract_surface()

    grid = grid.compute_cell_sizes()
    grid = grid.compute_normals()
    grid = grid.cell_data_to_point_data()

    dS = grid["Area"]
    n = grid["Normals"]

    E = grid["Er"] + 1j * grid["Ei"]
    H = grid["Hr"] + 1j * grid["Hi"]

    Sm = np.cross(E, np.conj(H), axis=-1)

    Savg = np.real(Sm)

    pwr = np.sum(Savg * dS[..., None] * n)
    print(f"{pwr:0.3f}")

    return pwr


# def integrate_power(ds: pv.DataSet):
#     grid = ds.extract_surface()
#     grid = grid.compute_normals(inplace=True)  # Point-based normals
#     grid = grid.compute_cell_sizes(
#         length=False, area=True, volume=False
#     )  # Point-based areas

#     # Convert everything to cell_data
#     grid = grid.point_data_to_cell_data()

#     dS = grid.cell_data["Area"]  # Now cell-based
#     n_cell = grid.cell_data["Normals"] / np.linalg.norm(
#         grid.cell_data["Normals"], axis=-1, keepdims=True
#     )  # Unit cell normals
#     E = grid.cell_data["Er"] + 1j * grid.cell_data["Ei"]
#     H = grid.cell_data["Hr"] + 1j * grid.cell_data["Hi"]

#     Sm = np.cross(E, np.conj(H)) / 2  # Time-averaged
#     Savg = np.real(Sm)

#     pwr = np.sum(Savg * n_cell * dS[:, None])  # Proper flux
#     print(pwr)
#     return pwr
