from glimmer import logclip
import quadpy

from dataclasses import dataclass, field

import numpy as np


import cupy as cp

from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres

import pyvista as pv

from scipy.constants import epsilon_0, mu_0, c

from glimmer.tools import area, rwg_connectivity

pv.global_theme.colorbar_orientation = "vertical"
pv.global_theme.cmap = "jet"

eta = cp.sqrt(mu_0 / epsilon_0)


@dataclass
class MoM(pv.PolyData):

    mesh: pv.UnstructuredGrid
    k: float

    probes: list = field(default_factory=list)

    deg: int = 2
    chunks: int = 16

    def __post_init__(self):

        # extract surface from unstructured grid
        poly = self.mesh.extract_surface()
        poly.triangulate(inplace=True)
        poly.clean()

        # extract surface from unstructured grid
        super().__init__(poly)

        # define angular frequency
        self.omega = self.k * c

        # define connectivity
        self.con = self.faces.reshape(-1, 4)[:, 1:]
        self.con_m, self.isin = rwg_connectivity(self.con)

        # number of edges
        self.M = self.con_m.shape[-2]

        # number of faces
        self.N = self.con.shape[-2]

        print(f"edges: {self.M}")
        print(f"faces: {self.N}")

        # face vertices (N x 3 x 3)
        self.r = cp.asarray(self.points[self.con])

        # rwg vertices (2 x M x 3 x 3)
        self.rm = cp.asarray(self.points[self.con_m])

        # rwg edge length
        self.lm = cp.linalg.norm(self.rm[0, :, 0] - self.rm[0, :, 1], axis=-1)

        # rwg centroids (2 x M x 3)
        self.rmc = cp.mean(self.rm, axis=-2)

        # rwg vertex fields (2 x 3 x M x 3)
        self.Em = cp.asarray(self["Er"] + 1j * self["Ei"])[self.con_m]

        # rwg centroid fields (2 x M x 3)
        self.Emc = cp.mean(self.Em, axis=-2)

        self.Am = area(
            self.rm[:, :, 0] - self.rm[:, :, -1], self.rm[:, :, 1] - self.rm[:, :, -1]
        )

        # areas of faces (N)
        self.dS = area(self.r[:, 1] - self.r[:, 0], self.r[:, 2] - self.r[:, 0])
        # self.dS = self.compute_cell_sizes()["Area"]

        # face centroids (N x 3)
        self.rc = np.mean(self.r, axis=-2)

    def solve(self):

        self.subdivide_by_quadrature()

        self.define_basis_functions()

        self.build_matrices()

        self.solve_currents()

    def post(self):

        for probe in self.probes:
            self.radiate(probe, self.chunks)

    def plot(self):

        plotter = pv.Plotter()

        for probe in self.probes:
            if probe.dimensionality == 3:
                distance = probe.length / np.linalg.norm(probe.dimensions)
                plotter.add_volume(probe, opacity_unit_distance=distance)
            else:
                plotter.add_mesh(probe)

        plotter.enable_parallel_projection()
        plotter.show_grid()

        return plotter

    def solve_currents(self):

        print("gmres solve basis currents")
        self.I, info = gmres(self.Z, self.V)

        if info > 0:
            RuntimeError("convergence to tolerance not achieved, number of iterations")
        else:
            print("converged !")

        print("integrating surface currents ...")
        J = np.sum(
            self.I[:, None, None, None] * self.fm * self.dSp[..., None], axis=(0, 1, -2)
        ).get()

        # Compute magnitude squared for plotting
        print("saving current vectors")
        self.cell_data["Jr"] = J.real
        self.cell_data["Ji"] = J.imag
        self.cell_data["|J|^2"] = np.linalg.norm(J, axis=-1) ** 2

    def build_matrices(self):

        print("interaction displacement vector (2 x M x M)")
        Rm = cp.linalg.norm(self.rmc[..., None, None, :] - self.rp, axis=-1)

        print("Greens function (2 x M x N) ...")
        G = cp.exp(-1j * self.k * Rm) / (4 * cp.pi * Rm)
        G[~cp.isfinite(G)] = 0

        # broadcast integration points
        GmdSp, dFpn = cp.broadcast_arrays(G * self.dSp, self.dfm[..., None])

        # reshape along integration points
        GmdSp = GmdSp.reshape(*GmdSp.shape[:2], -1)
        dFpn = dFpn.reshape(*dFpn.shape[:2], -1).transpose(0, 2, 1)
        Fpn = self.fm.reshape(*self.fm.shape[:2], -1, 3).transpose(0, 2, 1, 3)

        print("centroid basis displacement rwg centroid to free vertex ...")
        pm = cp.array([1, -1])[:, None, None]
        self.rhomc = pm * (self.rmc - self.rm[:, :, -1])

        # build impedance matrix
        # RHS = cp.sum(eta * Fpn * self.rhomc[:, None] / 2, axis=-1) - pm * dFpn / eta
        # Z0 = [1j * self.k * self.lm * (GmdSp[i] @ csr_matrix(RHS[i])) for i in range(2)]
        # self.Z = -(Z[0] + Z[1])

        print("computing electric potential ...")
        self.phi = cp.empty((2, self.M, self.M), dtype=cp.complex128)
        for i in range(2):
            self.phi[i] = 1j / (self.omega * epsilon_0) * GmdSp[i] @ csr_matrix(dFpn[i])

        print("computing magnetic vector potential ...")
        self.Avec = cp.empty((2, self.M, self.M, 3), dtype=cp.complex128)
        for i in range(2):
            for j in range(3):
                self.Avec[i, ..., j] = mu_0 * GmdSp[i] @ csr_matrix(Fpn[i, ..., j])

        print("forming excitation vector ...")
        self.V = self.lm * cp.sum(self.Emc * self.rhomc / 2, axis=(0, -1))

        # fix
        print("forming impedance matrix ...")
        self.Z = self.lm * (
            1j
            * self.omega
            * cp.sum(self.Avec * self.rhomc[..., None, :] / 2, axis=(0, -1))
            - cp.diff(self.phi, axis=0)[0]
        )

        # for Z in [self.Z0, self.Z, self.Z0 - self.Z]:
        #     Z[Z == 0] = np.nan
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     pcm1 = ax1.pcolormesh(np.real(Z.get()))
        #     pcm2 = ax2.pcolormesh(np.imag(Z.get()))
        #     fig.colorbar(pcm1, ax=ax1)
        #     fig.colorbar(pcm2, ax=ax2)

        # plt.show()

    def define_basis_functions(self):

        print("compute integration point to free vertex displacement  (2 x M x N x 3)")
        rho = self.rp - self.rm[..., -1, :][..., None, None, :]

        # match con to rwg connectivity (2 x M x N)
        pm = np.array([1, -1])[:, None, None]
        ind = cp.asarray(pm * self.isin)

        # consider summing along +/- axis, since this isn't in the notation technicaly specified
        print("computing rwg basis function")
        self.fm = (ind * (self.lm / self.Am)[..., None])[..., None, None] * rho

        print("computing rwg basic function divergence (2 x M x N)")
        self.dfm = ind * (self.lm / self.Am)[..., None]

    def subdivide_by_quadrature(self):
        print("define source points r' at by quadrature (N, n, 3)")
        scheme = quadpy.t2.get_good_scheme(self.deg)
        self.rp = cp.sum(
            self.r[..., None, :] * cp.asarray(scheme.points)[..., None], axis=1
        )
        self.dSp = self.dS[:, None] * cp.asarray(scheme.weights)

    def radiate(self, probe: pv.StructuredGrid, chunks: int = 1):
        """radiate current sources to probe"""

        r1 = cp.asarray(self.cell_centers().points)
        dSp = cp.asarray(self.compute_cell_sizes()["Area"])[..., None]

        J = cp.asarray(self.cell_data["Jr"] + 1j * self.cell_data["Ji"])

        divJr = self.compute_derivative("Jr", divergence=True)["divergence"]
        divJi = self.compute_derivative("Ji", divergence=True)["divergence"]
        divJ = cp.asarray(divJr + 1j * divJi)[..., None]

        def Es(r2):
            """radiate r1 onto chunk of r2, re-derive using green't function with positive exponent"""
            r = cp.asarray(r2)[..., None, :] - r1
            R = cp.linalg.norm(r, axis=-1, keepdims=True)
            G = cp.exp(-1j * self.k * R) / (4 * cp.pi * R)

            dG = r * (G / R**2 + 1j * self.k * G / R)

            A = mu_0 * cp.sum(J * G * dSp, axis=-2)
            grad_phi = cp.sum(1j * divJ / self.omega * dG * dSp, axis=-2)

            return (-1j * self.omega * A - grad_phi).get()

        print("radiating EFIE ...")
        E = np.concat([Es(r2) for r2 in np.array_split(probe.points, chunks)])

        probe.point_data["Er"] = E.real
        probe.point_data["Ei"] = E.imag
        E2 = np.linalg.norm(E, axis=-1) ** 2
        E2, E2dB = logclip(E2)
        probe.point_data["|E|^2"] = E2
        probe.point_data["|E|^2 (dB)"] = E2dB


import matplotlib.pyplot as plt
