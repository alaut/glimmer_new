from pyacvd import Clustering
from dataclasses import dataclass
import time
import numpy as np
import pyvista as pv


import cupy as cp

from . import eta


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

    text: str = "Working"

    def __enter__(self):
        self.start = time.time()
        print(self.text, end="\t")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[{elapsed:.1f} s]")


def integrate_power(ds: pv.DataSet, scale=1):
    """integrate E/H field along pyvista DataSet"""

    grid = ds.extract_surface()
    grid = grid.compute_cell_sizes()
    grid = grid.compute_normals()
    grid = grid.cell_data_to_point_data()

    dA = grid["Area"]
    n = grid["Normals"]
    E = grid["Er"] + 1j * grid["Ei"]

    try:
        H = grid["Hr"] + 1j * grid["Hi"]

        S = np.real(np.cross(E, np.conj(H)))
        pwr = np.sum(S * dA[..., None] * n) * scale
        print(f"power (E/H): {pwr:0.3f}")
    except:

        S = np.linalg.norm(E, axis=-1) ** 2 / eta
        pwr = np.sum(S * dA) * scale

        print(f"power (E): {pwr:0.3f}")

    return pwr
