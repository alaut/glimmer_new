import os
import time
import cupy as cp
import numpy as np

import pyvista as pv
from scipy.constants import mu_0, epsilon_0
import numpy as np

eta = np.sqrt(mu_0 / epsilon_0)


def logclip(ds: pv.DataSet, key: str, dBmin: float = -30, clip: float = 99):
    """process visualization dataset and re-represent in dB"""

    Amax = np.nanpercentile(ds[key], clip)
    ds[key] = np.clip(ds[key], 0, Amax)

    with np.errstate(divide="ignore", invalid="ignore"):
        ds[f"{key} (dB)"] = np.clip(10 * np.log10(ds[key] / Amax), dBmin, 0)


def set_fields(ds: pv.DataSet, E=None, H=None):

    ds["||E||^2"] = np.linalg.norm(E, axis=-1) ** 2
    ds["Er"] = np.real(E)
    ds["Ei"] = np.imag(E)

    ds["||H||^2"] = np.linalg.norm(H, axis=-1) ** 2
    ds["Hr"] = np.real(H)
    ds["Hi"] = np.imag(H)


def get_fields(ds: pv.DataSet):

    E = ds["Er"] + 1j * ds["Ei"]
    H = ds["Hr"] + 1j * ds["Hi"]

    return E, H


def add_fields(ds: pv.DataSet, E=None, H=None):

    E0, H0 = get_fields(ds)

    E += E0
    H += H0

    set_fields(ds, E, H)


def Gaussian(w0, lam, num_lam=3, num_waist=3, P0=1):
    """generate gaussian source object"""

    w0 = np.array([1, 1]) * np.array(w0)
    L = w0 * num_waist

    dL = lam / num_lam

    grid = RectGrid(L, dL)

    X = grid.points[..., 0] / w0[0]
    Y = grid.points[..., 1] / w0[1]

    I0 = 2 * P0 / (np.pi * w0[0] * w0[1])
    A = np.sqrt(I0 * eta) * np.exp(-(X**2)) * np.exp(-(Y**2))

    set_fields(
        grid,
        E=A[..., None] * np.array([1, 0, 0]),
        H=A[..., None] * np.array([0, 1, 0]) / eta,
    )

    grid._name = f"Gaussian w0=({w0[0]:0.2f}, {w0[1]:0.2f})"

    return grid


def RectGrid(L, dL):
    """generate rectangular grid"""

    L = np.array([1, 1]) * np.array(L)

    x = np.linspace(-L[0] / 2, L[0] / 2, int(L[0] / dL) + 1)
    y = np.linspace(-L[1] / 2, L[1] / 2, int(L[1] / dL) - 1)

    X, Y, Z = np.meshgrid(x, y, 0, indexing="ij")

    return pv.StructuredGrid(X, Y, Z)


def Mirror(L, dL, f=None, dz=0):
    """generate rectilinear mirror"""

    grid = RectGrid(L, dL)

    # apply parabolic deformation
    if f is not None:
        f = np.array([1, 1]) * np.array(f)
        x = grid.points[..., 0]
        y = grid.points[..., 1]
        grid.points[..., 2] += (x**2 / f[0] + y**2 / f[1]) / 4

    grid.points[..., 2] += dz
    grid._name = f"Mirror L=-({L[0]:0.2f}, {L[1]:0.2f}), f=({f[0]:0.2f}, {f[1]:0.2f})"

    set_fields(grid, E=np.zeros_like(grid.points), H=np.zeros_like(grid.points))

    return grid


def dot(A, B):
    return cp.sum(A * B, axis=-1, keepdims=True)


def cross(A, B):
    return cp.cross(A, B, axis=-1)


def norm(A):
    return cp.linalg.norm(A, axis=-1, keepdims=True)


def estimate_chunks(ds1: pv.DataSet, ds2: pv.DataSet, bytes=16, sf=25, verbose=False):
    """estimate needed chunking given mutual interaction of two pointclouds"""

    N1 = ds1.points.shape[0]
    N2 = ds2.points.shape[0]
    mem = cp.cuda.runtime.memGetInfo()[1]

    chunks = int(np.ceil(3 * N1 * N2 * bytes * sf / mem))
    if verbose:
        print(f"{N1} x {N2}->\t{chunks:03d} (sf={sf:0.3f})")

    return chunks


def radiate(k1: float, ds1: pv.DataSet, ds2: pv.DataSet, mode="radiate", chunks=None):
    """radiate E/H fields from source object to probe using Stratton-Chu equation"""

    start = time.time()

    tri = ds1.extract_surface()

    n1 = cp.asarray(tri.compute_normals(inplace=True, flip_normals=True)["Normals"])
    dA1 = cp.asarray(
        tri.compute_cell_sizes(length=False, area=True, volume=False)["Area"]
    )

    cells = tri.point_data_to_cell_data()

    E1 = cp.array(cells["Er"] + 1j * cells["Ei"])
    H1 = cp.array(cells["Hr"] + 1j * cells["Hi"])

    r1 = cp.asarray(tri.cell_centers().points)
    r2 = cp.asarray(ds2.points)

    match mode:
        case "reflect":
            E1 = 2 * dot(E1, n1) * n1 - E1
            H1 = H1 - 2 * dot(H1, n1) * n1
        case "negate":
            E1 = -E1
            H1 = -H1

    E2 = np.empty(r2.shape, dtype=complex)
    H2 = np.empty(r2.shape, dtype=complex)

    num_chunks = estimate_chunks(ds1, ds2) if chunks is None else chunks

    print(f"{mode}\t{ds1._name} onto {ds2._name} [{num_chunks} chunks]", end="")

    for ind in np.array_split(np.arange(r2.shape[0]), num_chunks):

        r = r2[ind, None, :] - r1

        R = norm(r)

        G = cp.exp(1j * k1 * R) / (4 * cp.pi * R)
        dG = r * (G / R**2 - 1j * k1 * G / R)

        Js = cross(n1, H1)
        Ms = cross(E1, n1)

        dE = cross(dG, Ms) + 1j * k1 * Js * G * eta + dot(n1, E1) * dG
        dH = cross(Js, dG) + 1j * k1 * Ms * G / eta + dot(n1, H1) * dG

        E2[ind] = cp.nansum(dA1[..., None] * dE, axis=-2).get()
        H2[ind] = cp.nansum(dA1[..., None] * dH, axis=-2).get()

    add_fields(ds2, E=E2, H=H2)

    print(f" in {time.time()-start:0.1f} s")


def reflect(*args, **kwargs):
    return radiate(*args, **kwargs, mode="reflect")


def negate(*args, **kwargs):
    return radiate(*args, **kwargs, mode="negate")


def Volume(d, xlim=0, ylim=0, zlim=0):
    """indexing needs to by ij if ray tracing is to work in ParaView"""

    if np.array(xlim).size == 2:
        xlim = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / d))

    if np.array(ylim).size == 2:
        ylim = np.linspace(ylim[0], ylim[1], int((ylim[1] - ylim[0]) / d))

    if np.array(zlim).size == 2:
        zlim = np.linspace(zlim[0], zlim[1], int((zlim[1] - zlim[0]) / d))

    X, Y, Z = np.meshgrid(xlim, ylim, zlim, indexing="ij")

    grid = pv.StructuredGrid(X, Y, Z)
    grid._name = f"Volume xlim=({xlim[0]:0.2f}, {xlim[-1]:0.2f}), ylim=({ylim[0]:0.2f}, {ylim[-1]:0.2f}), zlim=({zlim[0]:0.2f}, {zlim[-1]:0.2f})"

    set_fields(grid, E=np.zeros_like(grid.points), H=np.zeros_like(grid.points))

    return grid


def Plot(objects, plotter=None, cmap="jet"):

    start = time.time()

    if plotter is None:
        plotter = pv.Plotter()

    for obj in objects:
        if obj.dimensionality == 3:
            plotter.add_volume(obj, cmap=cmap)
        else:
            plotter.add_mesh(obj, cmap=cmap)

    print(f"plotted in {time.time()-start:0.1f} s")

    return plotter


def solve(lam, source, optics, probes, prefix=None):
    """physical optics algorithm"""

    k1 = 2 * np.pi / lam

    start = time.time()

    sources = [(source, radiate)]

    for optic in optics:
        src, fun = sources[-1]
        fun(k1, src, optic)
        sources.extend([(optic, negate), (optic, reflect)])

    for probe in probes:
        for src, fun in sources:
            fun(k1, src, probe)

    print(f"solved in {time.time()-start:0.1f} s")

    plotter = Plot([source, *optics, *probes])

    if prefix:

        start = time.time()

        os.makedirs(os.path.dirname(prefix), exist_ok=True)

        source.save(f"{prefix}.source.vtk")

        for i, optic in enumerate(optics):
            optic.save(f"{prefix}.optic.{i}.vtk")

        for i, probe in enumerate(probes):
            probe.save(f"{prefix}.probe.{i}.vtk")

        print(f"saved in {time.time()-start:0.1f} s")

    return plotter
