from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np

import pyvista as pv


from scipy.constants import mu_0, epsilon_0, c


eta = np.sqrt(mu_0 / epsilon_0)


def set_field(ds, A, key, dtype=np.float32):
    ds[f"{key}r"] = np.real(A).astype(dtype)
    ds[f"{key}i"] = np.imag(A).astype(dtype)


def get_field(ds, key):
    """retrieve named field component from pyvista DataSet"""
    Ar = ds[f"{key}r"] if f"{key}r" in ds.array_names else np.zeros_like(ds.points)
    Ai = ds[f"{key}i"] if f"{key}i" in ds.array_names else np.zeros_like(ds.points)
    return Ar + 1j * Ai


def get_fields(ds: pv.DataSet, keys=["E", "H"]):
    """retrieve E/H complex vector fields from pyvista DataSet"""
    return [get_field(ds, key) for key in keys]


def add_field(ds: pv.DataSet, A=0, key="E"):

    A0 = get_field(ds, key)

    set_field(ds, A + A0, key)


def add_fields(ds: pv.DataSet, E=None, H=None, J=None):
    """add E/H complex vector fields to pyvista DataSet"""

    if E is not None:
        add_field(ds, E, "E")

    if H is not None:
        add_field(ds, H, "H")

    if J is not None:
        add_field(ds, J, "J")


def process_fields(ds: pv.DataSet, keys=["E", "H"], clip=99, dBmin=-30):
    """process complex vector fields as scalar field amplitude"""

    for key in keys:
        A = get_field(ds, key)

        I = np.linalg.norm(A, axis=-1) ** 2
        I = np.clip(I, max=np.nanpercentile(I, clip))

        IdB = np.clip(10 * np.log10(I / I.max()), dBmin, 0)

        ds[f"||{key}||^2"] = I
        ds[f"||{key}||^2 (dB)"] = IdB


def Mirror(L, dL, f=None, dz=0):
    """generate rectilinear mirror with parabolic deformation"""

    Lx, Ly = np.array([1, 1]) * np.array(L)
    grid = Grid(d=dL, xlim=(-Lx / 2, Lx / 2), ylim=(-Ly / 2, Ly / 2))

    # apply parabolic deformation
    if f is not None:
        fx, fy = np.array([1, 1]) * np.array(f)
        x = grid.points[..., 0]
        y = grid.points[..., 1]
        grid.points[..., 2] += -(x**2 / fx + y**2 / fy) / 4

    grid.points[..., 2] += dz

    return grid


def Grid(d, xlim=None, ylim=None, zlim=None, ds: pv.DataSet = None, scale=1.0):
    """yields structured Plane/Grid given bounds or bounding DataSet"""

    if ds is not None:

        ds = ds.scale(scale, point=ds.center)

        xmin, xmax, ymin, ymax, zmin, zmax = ds.bounds

        if xlim is None:
            xlim = (xmin, xmax)

        if ylim is None:
            ylim = (ymin, ymax)

        if zlim is None:
            zlim = (zmin, zmax)

    if xlim is None:
        xlim = 0
    if ylim is None:
        ylim = 0
    if zlim is None:
        zlim = 0

    if np.array(xlim).size == 2:
        xlim = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / d))

    if np.array(ylim).size == 2:
        ylim = np.linspace(ylim[0], ylim[1], int((ylim[1] - ylim[0]) / d))

    if np.array(zlim).size == 2:
        zlim = np.linspace(zlim[0], zlim[1], int((zlim[1] - zlim[0]) / d))

    X, Y, Z = np.meshgrid(xlim, ylim, zlim, indexing="ij")

    return pv.StructuredGrid(X.astype(float), Y.astype(float), Z.astype(float))


def Plot(objects, plotter=None, cmap="jet"):

    def plot(ds):

        try:
            distance = ds.length / np.linalg.norm(ds.dimensions)
            actor = plotter.add_volume(ds, cmap=cmap, opacity_unit_distance=distance)
        except:
            actor = plotter.add_mesh(ds, cmap=cmap)

        return actor

    with Timer("Plotting"):

        pv.global_theme.cmap = cmap

        if plotter is None:
            plotter = pv.Plotter()

        actors = [plot(ds) for ds in objects]

    plotter.enable_parallel_projection()
    plotter.show_grid()

    return plotter


@dataclass
class Timer:

    message: str = "Working"

    stop: datetime = None
    start: datetime = None
    elapsed: timedelta = None

    def __enter__(self):
        self.start = datetime.now()
        print(self.message, end="\t")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = datetime.now()
        self.elapsed = self.stop - self.start
        print(f"[{self.elapsed.total_seconds():.1f} s]")
