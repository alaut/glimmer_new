import os
from dataclasses import dataclass, field
import quadpy
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres
import pyvista as pv

import cupy as cp
import numpy as np


from . import (
    mu_0,
    epsilon_0,
    c,
    Plot,
    get_field,
    process_fields,
    set_field,
    add_field,
)
from .tools import Timer, remesh


@dataclass
class Solver:

    ds: pv.DataSet
    lam: float

    probes: list = field(default_factory=list)

    chunks: int = 32
    tolerance: float = 1e-6

    remesh: float = None

    def __post_init__(self):

        self.tri = self.ds.extract_surface()
        self.tri.triangulate(inplace=True)
        self.tri.clean(inplace=True, tolerance=self.tolerance)

        if self.remesh is not None:
            self.tri = remesh(self.tri, self.remesh)

    def solve(self):
        """Solve EFIE on PEC triangulation with method of moments."""

        k = 2 * np.pi / self.lam
        omega = k * c

        with Timer("Build Face Connectivity"):
            con = get_connectivity(self.tri)

        with Timer("Generate RWG connectivity"):
            con_m, isin = rwg(con)

        with Timer("Subdivide faces by quadrature"):
            rp, dSp = subdivide_faces_by_quadrature(self.tri, con)

        with Timer("Compute RWG data"):
            rm, lm, rmc, Am, Emc, rhomc = get_rwg_geometry(self.tri, con_m)

        with Timer("Build RWG basis functions"):
            fm, dfm = build_basis_functions(rp, rm, lm, Am, isin)

        with Timer("Generate Green's Function"):
            Gm = green_function(rmc, rp, k)

        with Timer("Assemble scalar potential matrices"):
            phi = assemble_scalar_potential(Gm * dSp, dfm, omega)

        with Timer("Assemble vector potential matrices"):
            Avec = assemble_vector_potential(Gm * dSp, fm)

        with Timer("Build excitation vector"):
            V = excitation_vector(lm, Emc, rhomc)

        with Timer("Build impedance matrix"):
            Z = impedance_matrix(lm, omega, Avec, rhomc, phi)

        with Timer("solve currents GMRES"):
            I = solve_currents(Z, V)

        with Timer("solve sources by integrate currents"):
            J = integrate_currents(I, fm, dSp)

        with Timer("setting fields"):
            set_field(self.tri, A=J.get(), key="J")
            process_fields(self.tri, keys="J")
            self.tri.set_active_scalars("||J||^2")

        for probe in self.probes:
            radiate(k, self.tri, probe, self.chunks)
            process_fields(probe, keys="E")
            probe.set_active_scalars("||E||^2")

    def plot(self):

        plotter = Plot([self.tri, *self.probes])

        return plotter

    def save(self, prefix):
        with Timer("saving ..."):

            os.makedirs(os.path.dirname(prefix), exist_ok=True)

            for i, obj in enumerate([self.ds, self.tri, *self.probes]):
                obj.save(f"{prefix}.{i:03d}.vtk")


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

    c1p = np.stack([internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1)
    c1n = np.stack([internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1)

    # define rwg connectivity
    con_m = np.stack([c1p, c1n])

    # define adjacency
    isin = np.all(np.sort(con) == np.sort(con_m)[:, :, None], axis=-1)
    r_in_T = np.array([1, -1])[:, None, None] * isin

    return con_m, r_in_T


def get_connectivity(tri):
    """return triangulation connectivity"""
    return tri.faces.reshape(-1, 4)[:, 1:]


def subdivide_faces_by_quadrature(tri: pv.PolyData, con, deg: int = 3, show=False):
    """subdivide triangulation for integration by quadrature"""

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
    u = rm[0] - rm[-1]
    v = rm[1] - rm[-1]
    Am = 0.5 * cp.linalg.norm(cp.cross(u, v, axis=-1), axis=-1)

    # RWG fields (at vertices, centroids)
    Em = cp.asarray(get_field(tri, "E")[con_m])
    # Em = cp.asarray(get_field(tri, "E")[con_m])
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

    # basis function
    fm = 0.5 * dfm[..., None, None] * rho

    return fm, dfm


def green_function(rmc, rp, k):
    """Return Green's function at quadrature points"""
    Rm = cp.linalg.norm(cp.asarray(rmc[..., None, None, :]) - cp.asarray(rp), axis=-1)
    Gm = cp.exp(-1j * k * Rm) / Rm
    # G[~cp.isfinite(G)] = 0
    return Gm


def assemble_scalar_potential(GmdS, dfm, omega):
    """Rao 1982 eq. 20"""

    _, M, N, n = GmdS.shape

    GdSm, dfm = cp.broadcast_arrays(GmdS, dfm[..., None])

    GdSmp = GdSm.reshape(2, M, -1)
    dfpn = dfm.reshape(2, M, -1).transpose(0, 2, 1)

    phi = cp.empty((2, M, M), dtype=cp.complex128)
    for i in range(2):
        phi[i] = GdSmp[i] @ csr_matrix(dfpn[i])

    # eq. 20
    return -1 / (4 * np.pi * 1j * omega * epsilon_0) * phi


def assemble_vector_potential(GmdS, fm):
    """Rao 1982 eq. 19"""

    _, M, N, n = GmdS.shape

    GdSmp = GmdS.reshape(2, M, -1)
    fpn = fm.reshape(2, M, -1, 3).transpose(0, 2, 1, 3)

    A = cp.empty((2, M, M, 3), dtype=cp.complex128)
    for i in range(2):
        for j in range(3):
            A[i, ..., j] = GdSmp[i] @ csr_matrix(fpn[i, ..., j])

    return mu_0 / (4 * np.pi) * A


def excitation_vector(lm, Emc, rhomc):
    """Rao 1982 eq. 18"""
    return lm * np.sum(Emc * rhomc / 2, axis=(0, -1))


def impedance_matrix(lm, omega, Avec, rhomc, phi):

    Z = lm * (
        1j * omega * cp.sum(Avec * rhomc[..., None, :] / 2, axis=(0, -1))
        + phi[0]
        - phi[1]
    )

    return Z


def solve_currents(Z, V):
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

    divJr = ds1.compute_derivative("Jr", divergence=True)["divergence"]
    divJi = ds1.compute_derivative("Ji", divergence=True)["divergence"]
    divJ = divJr + 1j * divJi

    r2 = ds2.points

    # print()
    with Timer(f"radiating EFIE int {chunks} chunks..."):
        E = np.empty_like(r2, dtype=complex)
        for ind in np.array_split(np.arange(r2.shape[0]), chunks):
            E[ind] = efie(k1, r1, dS1, J, divJ, r2[ind])

    add_field(ds2, E, "E")


def efie(k1, r1, dS1, J1, divJ1, r2):
    """radiate r1 onto chunk of r2, re-derive using green't function with positive exponent"""

    omega = k1 * c

    r = cp.asarray(r2)[..., None, :] - cp.asarray(r1)
    R = cp.linalg.norm(r, axis=-1, keepdims=True)
    G = cp.exp(-1j * k1 * R) / R

    dGp = G * (1 + 1j * k1 * R) * r / R**2

    dS1 = cp.asarray(dS1)[..., None]
    J1 = cp.asarray(J1)
    divJ1 = cp.asarray(divJ1)[..., None]

    # eq. 4
    sigma = -divJ1 / (1j * omega)

    # eq. 3
    grad_phi = 1 / (4 * np.pi * epsilon_0) * cp.sum(sigma * dGp * dS1, axis=-2)

    # eq. 2
    A = mu_0 / (4 * np.pi) * cp.sum(J1 * G * dS1, axis=-2)

    # eq. 1
    Es = -1j * omega * A - grad_phi

    return Es.get()
