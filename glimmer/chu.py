from dataclasses import dataclass
import os
import cupy as cp
import pyvista as pv
import numpy as np

from .tools import Timer
from . import add_fields, eta, Plot, process_fields


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

    available, total = cp.cuda.runtime.memGetInfo()

    # print(f"GPU: {available/total*100:0.1f}%")

    chunks = int(np.ceil(3 * N1 * N2 * bytes * sf / available))
    if verbose:
        print(f"{N1} x {N2}->\t{chunks:03d} (sf={sf:0.3f})")

    return chunks


def radiate(k1: float, ds1: pv.DataSet, ds2: pv.DataSet, mode="radiate", chunks=None):
    """radiate E/H fields from source to probe using Stratton-Chu equation"""

    poly = ds1.extract_surface()
    poly = poly.compute_cell_sizes()
    poly = poly.compute_normals()
    poly = poly.cell_data_to_point_data()

    r1 = cp.array(poly.points)
    E1 = cp.array(poly["Er"] + 1j * poly["Ei"])
    H1 = cp.array(poly["Hr"] + 1j * poly["Hi"])

    dA1 = cp.asarray(poly["Area"]) * (1 + poly.is_all_triangles)
    n1 = cp.asarray(poly["Normals"])

    match mode:
        case "reflect":
            E1 = 2 * dot(E1, n1) * n1 - E1
            H1 = H1 - 2 * dot(H1, n1) * n1
        case "negate":
            E1 = -E1
            H1 = -H1

    r2 = cp.asarray(ds2.points)
    E2 = np.empty(r2.shape, dtype=complex)
    H2 = np.empty(r2.shape, dtype=complex)

    num_chunks = estimate_chunks(ds1, ds2) if chunks is None else chunks

    with Timer(f"{mode}\t{ds1.points.shape} onto {ds2.points.shape}"):

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


def reflect(*args, **kwargs):
    return radiate(*args, **kwargs, mode="reflect")


def negate(*args, **kwargs):
    return radiate(*args, **kwargs, mode="negate")


@dataclass
class Solver:
    """Stratton-Chu physical optics solver"""

    lam: float

    source: pv.DataSet
    optics: list = None
    probes: list = None

    def solve(self):
        """radiate source through optics and onto probes"""

        with Timer("Solving ..."):

            k = 2 * np.pi / self.lam

            sources = [(self.source, radiate)]

            for optic in self.optics:
                src, fun = sources[-1]
                fun(k, src, optic)
                sources.extend([(optic, negate), (optic, reflect)])

            for probe in self.probes:
                for src, fun in sources:
                    fun(k, src, probe)

            for obj in [self.source, *self.optics, *self.probes]:
                process_fields(obj)
                obj.set_active_scalars("||E||^2")

    def save(self, prefix):
        """save pyvista objects to vtk format"""

        with Timer("saving ..."):

            os.makedirs(os.path.dirname(prefix), exist_ok=True)

            mb = pv.MultiBlock([self.source, *self.optics, *self.probes])
            mb.save(f"{prefix}.vtm")

            self.source.save(f"{prefix}.source.vtk")

            for i, optic in enumerate(self.optics):
                optic.save(f"{prefix}.optic.{i}.vtk")

            for i, probe in enumerate(self.probes):
                probe.save(f"{prefix}.probe.{i}.vtk")

    def plot(self):

        plotter = Plot([self.source, *self.optics, *self.probes])

        return plotter
