import sparse
import torch
import tensorflow as tf
import os
from dataclasses import dataclass, field
import quadpy
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres
import pyvista as pv

import cupy as cp
import numpy as np

from . import *

from .tools import remesh


@dataclass
class Solver:

    ds: pv.DataSet
    lam: float

    probes: list = field(default_factory=list)

    chunks: int = 32

    tolerance: float = None
    remesh: float = None

    mode: str = "cupy"

    def __post_init__(self):

        self.tri = self.ds.extract_surface()
        self.tri.triangulate(inplace=True)

        if self.tolerance is not None:
            self.tri.clean(inplace=True, tolerance=self.tolerance)

        if self.remesh is not None:
            self.tri = remesh(self.tri, self.remesh)

    def solve(self):
        """Solve EFIE on PEC triangulation with method of moments."""

        k = 2 * np.pi / self.lam
        omega = k * c

        print(f"Array Mode [{self.mode}]")

        with Timer("Building Face Connectivity"):
            con = get_connectivity(self.tri)

        with Timer("Generating RWG connectivity"):
            con_m, isin = rwg(con)

        with Timer("Subdividing faces by quadrature"):
            rp, dSp = subdivide_faces_by_quadrature(self.tri, con)

        with Timer("Computing RWG basis data"):
            rm, lm, rmc, Am, Emc, rhomc = get_rwg_geometry(self.tri, con_m)

        with Timer("Building RWG basis functions"):
            fm, dfm = build_basis_functions(rp, rm, lm, Am, isin)

        with Timer("Generating Green's Function"):
            Gm = green_function(rmc, rp, k)

        with Timer("Assembling scalar potentials"):
            phi = assemble_scalar_potential(Gm * dSp, 0 * dfm, omega, self.mode)

        with Timer("Assembling vector potentials"):
            Avec = assemble_vector_potential(Gm * dSp, fm, self.mode)

        with Timer("Building excitation vector"):
            V = excitation_vector(lm, Emc, rhomc)

        with Timer("Building impedance matrix"):
            Z = impedance_matrix(lm, omega, Avec, rhomc, phi)

        with Timer("Solving coefficients (GMRES)"):
            I = solve_currents(Z, V)

        with Timer("Integrating coefficients"):
            J = integrate_currents(I, fm, dSp)

        with Timer("Setting surface currents"):
            set_field(self.tri, A=J.get(), key="J")

            process_fields(self.tri, keys="J")
            self.tri.set_active_scalars("||J||^2")

            self.tri = self.tri.compute_derivative("Jr", divergence="divJr")
            self.tri = self.tri.compute_derivative("Ji", divergence="divJi")

        for probe in self.probes:
            radiate(k, self.tri, probe, self.chunks)
            process_fields(probe, keys="E")
            probe.set_active_scalars("||E||^2")

    def save(self, prefix):

        with Timer("Saving"):

            os.makedirs(os.path.dirname(prefix), exist_ok=True)

            mb = pv.MultiBlock([*self.probes, self.tri])
            mb.save(f"{prefix}.vtm")

    def plot(self):

        plotter = Plot([self.tri, *self.probes])

        return plotter


def get_free_vert(edge, edges):
    """find free vertex given edge (2,) in face edges (M, 3, 2)"""

    # match edge to face
    face = np.any(np.all(edge == edges, axis=-1), axis=-1)

    # get face to points
    points = np.unique(edges[face])

    # find free vertex
    vertex = np.setdiff1d(points, edge)

    return int(vertex[0])


def rwg(con):
    """generate RWG (Rao, Wilton, Glisson) connectivity from trimesh face connectivity"""

    # construct edges from face connectivity (face x edge x point) (N, 3, 2)
    edges = con[:, [[0, 1], [1, 2], [2, 0]]]

    # sort flattened edges by points (N x 3, 2)
    sorted_edges = np.sort(edges.reshape(-1, 2))

    # count edge uniqueness
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)

    # define rwg basis of unique internal edges (M, 2)
    internal_edges = unique_edges[counts == 2]

    # find free rwg vertices
    vert_pos = [get_free_vert(edge, edges) for edge in internal_edges]
    vert_neg = [get_free_vert(edge[::-1], edges) for edge in internal_edges]

    c1p = np.stack([internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1)
    c1n = np.stack([internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1)

    # define rwg connectivity
    con_m = np.stack([c1p, c1n])

    # define adjacency
    r_in_T = np.all(np.sort(con) == np.sort(con_m)[:, :, None], axis=-1)

    return con_m, r_in_T


def get_connectivity(tri):
    """return triangulation connectivity"""
    return tri.faces.reshape(-1, 4)[:, 1:]


def subdivide_faces_by_quadrature(tri: pv.PolyData, con, deg: int = 2, show=False):
    """subdivide triangulation for integration by quadrature (assert deg = 2, 3, 4, 6, 7, 12)"""

    r = tri.points[con]
    dS = tri.compute_cell_sizes(length=False, volume=False)["Area"]

    scheme = quadpy.t2.get_good_scheme(deg)
    rp = np.sum(r[..., None, :] * scheme.points[..., None], axis=1)
    dSp = dS[:, None] * scheme.weights

    if show:
        rc = tri.cell_centers()
        plotter = pv.Plotter()
        plotter.add_mesh(tri, style="wireframe")
        plotter.add_points(rp.reshape(-1, 3))
        plotter.add_points(rc, color="r", style="points_gaussian", point_size=0.5)
        plotter.show()

    return cp.array(rp), cp.array(dSp)


def get_rwg_geometry(tri, con_m):
    """Return RWG geometry"""

    # re-order for convenience ((v1p, v1n), (v2p, v2n), (vp, vn))
    con_m = con_m.transpose(2, 0, 1)

    # RWG points vertices
    rm = cp.asarray(tri.points[con_m])

    # RWG shared edge length
    lm = cp.linalg.norm(rm[0] - rm[1], axis=-1)[0]

    # RWG centroids
    rmc = cp.mean(rm, axis=0)

    # RWG areas
    u = rm[0] - rm[-1]  # v1 - v
    v = rm[1] - rm[-1]  # v2 - v
    Am = 0.5 * cp.linalg.norm(cp.cross(u, v, axis=-1), axis=-1)
    Am[-1] = -Am[-1]  # signed area (Alex Laut)

    # RWG fields at vertices
    Em = cp.asarray(get_field(tri, "E")[con_m])

    # RWG fields at centroids)
    Emc = cp.mean(Em, axis=0)

    # RWG centroid to vertex
    rhomc = cp.stack([rmc[0] - rm[-1, 0], rm[-1, 1] - rmc[1]])

    return rm, lm, rmc, Am, Emc, rhomc


def build_basis_functions(rp, rm, lm, Am, r_in_T):
    """RWG basis function and its divergence"""

    # displacement from integration point to free-vertex
    rho = rp - rm[-1][..., None, None, :]

    # basis function divergence
    dfm = cp.array(r_in_T) * (lm / Am)[..., None]
    dfm[~r_in_T] = 0

    # basis function
    fm = 0.5 * dfm[..., None, None] * rho
    fm[~r_in_T] = 0

    return fm, dfm


def green_function(rmc, rp, k):
    """Return Green's function at quadrature points"""
    Rm = cp.linalg.norm(cp.asarray(rmc[..., None, None, :]) - cp.asarray(rp), axis=-1)
    Gm = cp.exp(-1j * k * Rm) / (4 * np.pi * Rm)
    return Gm


def assemble_scalar_potential(GmdS, dfm, omega, mode="cupy"):
    """Rao 1982 eq. 20"""

    subscripts = "impq,inp->imn"

    match mode:
        case "cupy":
            phi = cp.einsum(subscripts, GmdS, dfm)
        case "cupy-sparse":
            _, M, N, n = GmdS.shape
            GdSm, dfm = cp.broadcast_arrays(GmdS, dfm[..., None])
            GdSmp = GdSm.reshape(2, M, -1)
            dfpn = dfm.reshape(2, M, -1).transpose(0, 2, 1)
            phi = cp.empty((2, M, M), dtype=cp.complex128)
            for i in range(2):
                phi[i] = GdSmp[i] @ csr_matrix(dfpn[i])
        case "numpy":
            phi = np.einsum(subscripts, GmdS.get(), dfm.get())
        case "numpy-sparse":
            dfm = sparse.as_coo(dfm.get())
            phi = cp.array(np.einsum(subscripts, GmdS.get(), dfm).todense())
        case "tensorflow":
            GmdS = tf.convert_to_tensor(GmdS.get(), dtype=tf.complex64)
            dfm = tf.convert_to_tensor(dfm.get(), dtype=tf.complex64)
            phi = cp.array(tf.einsum(subscripts, GmdS, dfm).numpy())
        case "pytorch":
            GmdS = torch.from_numpy(GmdS.get()).to(dtype=torch.complex64)
            dfm = torch.from_numpy(dfm.get()).to(dtype=torch.complex64)
            phi = torch.einsum(subscripts, GmdS, dfm)
            phi = cp.array(phi.numpy())

    return 1j / (omega * epsilon_0) * phi


def assemble_vector_potential(GmdS, fm, mode="cupy"):
    """Rao 1982 eq. 19"""

    subscripts = "impq,inpqj->imnj"

    match mode:
        case "cupy":
            A = cp.einsum(subscripts, GmdS, fm)
        case "cupy-sparse":
            _, M, N, n = GmdS.shape
            GdSmp = GmdS.reshape(2, M, -1)
            fpn = fm.reshape(2, M, -1, 3).transpose(0, 2, 1, 3)
            A = cp.empty((2, M, M, 3), dtype=cp.complex128)
            for i in range(2):
                for j in range(3):
                    A[i, ..., j] = GdSmp[i] @ csr_matrix(fpn[i, ..., j])
        case "numpy":
            A = np.array(np.einsum(subscripts, GmdS.get(), fm.get()))
        case "numpy-sparse":
            fm = sparse.asarray(fm.get())
            A = cp.array(np.einsum(subscripts, GmdS.get(), fm).todense())
        case "tensorflow":
            GmdS = tf.convert_to_tensor(GmdS.get(), dtype=tf.complex64)
            fm = tf.convert_to_tensor(fm.get(), dtype=tf.complex64)
            A = cp.array(tf.einsum(subscripts, GmdS, fm).numpy())
        case "pytorch":
            GmdS = torch.from_numpy(GmdS.get()).to(dtype=torch.complex64)
            fm = torch.from_numpy(fm.get()).to(dtype=torch.complex64)
            A = torch.einsum(subscripts, GmdS, fm)
            A = cp.array(A.numpy())

    return mu_0 * A


def excitation_vector(lm, Emc, rhomc):
    """Rao 1982 eq. 18"""
    return lm * np.sum(Emc * rhomc / 2, axis=(0, -1))


def impedance_matrix(lm, omega, Avec, rhomc, phi):
    """Rao 1982 eq. 17"""

    Z = lm[:, None] * (
        1j * omega * cp.sum(Avec * rhomc[..., None, :] / 2, axis=(0, -1))
        + phi[0]
        - phi[1]
    )

    return Z


def solve_currents(Z, V):
    """Rao 1982 eq. 16"""
    I, info = gmres(Z, V)
    if info > 0:
        raise RuntimeError("GMRES did not converge")
    return I


def integrate_currents(I, fm, dSp):
    """Rao 1982 eq. 9"""
    return cp.sum(I[:, None, None, None] * fm * dSp[..., None], axis=(0, 1, -2))


def radiate(k1: float, ds1: pv.DataSet, ds2: pv.DataSet, chunks: int = 8):
    """radiate current sources to probe"""

    r1 = ds1.cell_centers().points
    dS1 = ds1.compute_cell_sizes()["Area"]

    J = get_field(ds1, "J")
    divJ = get_field(ds1, "divJ")

    r2 = ds2.points

    with Timer(f"Radiating EFIE into {chunks} chunks"):
        E = np.empty_like(r2, dtype=complex)
        for ind in np.array_split(np.arange(r2.shape[0]), chunks):
            E[ind] = efie(k1, r1, dS1, J, divJ, r2[ind])

    add_field(ds2, E, "E")


def efie(k1, r1, dS1, J1, divJ1, r2):
    """radiate r1 onto chunk of r2, re-derive using green't function with positive exponent"""

    omega = k1 * c

    r = cp.asarray(r2)[..., None, :] - cp.asarray(r1)
    R = cp.linalg.norm(r, axis=-1, keepdims=True)
    G = cp.exp(-1j * k1 * R) / (4 * cp.pi * R)

    dGp = G * (1 + 1j * k1 * R) * r / R**2

    dS1 = cp.asarray(dS1)[..., None]
    J1 = cp.asarray(J1)
    divJ1 = cp.asarray(divJ1)[..., None]

    # eq. 4
    sigma = 1j * divJ1 / omega

    # eq. 3
    grad_phi = cp.sum(sigma * dGp * dS1, axis=-2) / epsilon_0

    # eq. 2
    A = mu_0 * cp.sum(J1 * G * dS1, axis=-2)

    # eq. 1
    Es = -1j * omega * A - grad_phi

    return Es.get()
