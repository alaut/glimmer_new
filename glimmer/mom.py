import quadpy
import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres


import pyvista as pv

from scipy.constants import epsilon_0, mu_0, c, milli, giga
from glimmer.rwg import setup_rwg_connectivity

pv.global_theme.colorbar_orientation = "vertical"


@dataclass
class MoM(pv.PolyData):

    mesh: pv.UnstructuredGrid
    k: float

    deg: int = 2

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
        self.con_m = setup_rwg_connectivity(self.con)

        # face vertices (N x 3 x 3)
        self.r = self.points[self.con]

        # rwg vertices (2 x M x 3 x 3)
        self.rm = self.points[self.con_m]

        # rwg edge length
        self.lm = np.linalg.norm(self.rm[0, :, 0] - self.rm[0, :, 1], axis=-1)

        # rwg centroids (2 x M x 3)
        self.rmc = np.mean(self.rm, axis=-2)

        # rwg vertex fields (2 x 3 x M x 3)
        self.Em = (self["Er"] + 1j * self["Ei"])[self.con_m]

        # rwg centroid fields (2 x M x 3)
        self.Emc = np.mean(self.Em, axis=-2)

        self.Am = area(
            self.rm[:, :, 0] - self.rm[:, :, -1], self.rm[:, :, 1] - self.rm[:, :, -1]
        )

        # areas of faces (N)
        self.dS = area(self.r[:, 1] - self.r[:, 0], self.r[:, 2] - self.r[:, 0])

        # face centroids (N x 3)
        self.rc = np.mean(self.r, axis=-2)

        # define plus/minus modifier (used to correct rho vector orientation)"
        pm = np.array([1, -1])[:, None, None]

        # match con to rwg connectivity (2 x M x N)
        self.ind = pm * np.all(
            np.sort(self.con) == np.sort(self.con_m)[:, :, None], axis=-1
        )

        # define centroid to free vertex displacement vector
        self.rhomc = pm * (self.rmc - self.rm[:, :, -1])

        # define integration points
        self.subdivide_faces()

        # define rwg basis functions
        self.define_basis_functions()

    def solve(self):

        self.compute_potentials()

        self.build_matrices()

        self.solve_currents()

    def solve_currents(self):

        print("gmres solve basis currents")

        Z = cp.asarray(self.Z)
        V = cp.asarray(self.V)

        I, info = gmres(Z, V)

        if info > 0:
            RuntimeError("convergence to tolerance not achieved, number of iterations")
        else:
            print("converged !")

        self.I = I.get()

        # solve for unknown surface currents
        print("integrating surface currents ...")
        J = np.sum(
            self.I[:, None, None, None] * self.fm * self.dSp[..., None], axis=(0, 1, -2)
        )

        # Compute magnitude squared for plotting
        print("saving current vectors")
        self.cell_data["Jr"] = J.real
        self.cell_data["Ji"] = J.imag
        self.cell_data["|J|^2"] = np.linalg.norm(J, axis=-1)

    def build_matrices(self):
        print("centroid basis displacement rwg centroid to free vertex ...")

        print("forming excitation vector ...")
        self.V = self.lm * np.sum(self.Emc * self.rhomc / 2, axis=(0, -1))

        print("forming impedance matrix ...")
        self.Z = self.lm * (
            1j
            * self.omega
            * np.sum(self.Avec * self.rhomc[:, :, None, :] / 2, axis=(0, -1))
            + (self.phi[0] - self.phi[1])
        )

    def define_basis_functions(self):

        print("compute integration point to free vertex displacement  (2 x M x N x 3)")
        rho = self.rp - self.rm[..., -1, :][..., None, None, :]

        # consider summing along +/- axis, since this isn't in the notation technicaly specified
        print("computing rwg basis function")
        self.fm = (self.ind * (self.lm / self.Am)[..., None])[..., None, None] * rho

        print("computing rwg basic function divergence (2 x M x N)")
        self.dfm = self.ind * (self.lm / self.Am)[..., None]

    def subdivide_faces(self):

        print("define source points r' at by quadrature (N, n, 3)")
        scheme = quadpy.t2.get_good_scheme(self.deg)
        self.rp = np.sum(self.r[..., None, :] * scheme.points[..., None], axis=1)
        self.dSp = self.dS[:, None] * scheme.weights

    def plot_excitation(self):

        self["Sr"] = np.cross(self["Er"], self["Hr"], axis=-1)
        scale = np.mean(self.lm) / np.max(np.linalg.norm(self["Er"], axis=-1)) / 2

        plotter = pv.Plotter()
        plotter.add_mesh(self, scalars="Er", cmap="jet", show_edges=True)
        plotter.add_mesh(self.glyph("Er", factor=scale), color="blue")
        plotter.add_mesh(self.glyph("Hr", factor=scale), color="red")
        plotter.add_mesh(self.glyph("Sr", factor=scale), color="green")
        plotter.enable_parallel_projection()
        plotter.show()

    def plot_rwg_basis(self):

        rp, rn = self.rm[:, :, -1]
        rhomcp, rhomcn = self.rhomc

        plotter = pv.Plotter()
        plotter.add_mesh(self, show_edges=True)
        plotter.add_arrows(rp, rhomcp, color="red")
        plotter.add_arrows(rn - rhomcn, rhomcn, color="blue")
        plotter.add_points(self.rp.reshape(-1, 3))
        plotter.add_point_labels(self.points, range(self.points.shape[0]))
        plotter.add_point_labels(self.rc, [str(con) for con in self.con])
        labels = [[str(x[:2]) for x in con_m] for con_m in self.con_m]
        plotter.add_point_labels(rp + rhomcp / 2, labels[0], text_color="red")
        plotter.add_point_labels(rn - rhomcn / 2, labels[1], text_color="blue")
        plotter.enable_parallel_projection()
        plotter.show()

    def plot_currents(self, lines=False, scalars="|J|^2"):

        print("plotting currents ...")

        plotter = pv.Plotter()
        plotter.add_mesh(self, scalars=scalars, show_edges=True, cmap="jet")

        if lines:
            i, j, k = np.indices(self.Rm.shape)
            i, j, k = np.nonzero(self.ind)
            lines = plotter.add_lines(
                lines=np.stack([self.rp[k], self.rmc[i, j]]).reshape(-1, 3, order="F"),
                width=0.1,
                color="black",
            )

        plotter.enable_parallel_projection()
        plotter.show_grid()
        plotter.show()

    def show_charts(self):

        plots = {}
        for k, v in self.data.items():

            if np.iscomplexobj(v):
                plots[f"Re({k})"] = np.real(v)
                plots[f"Im({k})"] = np.imag(v)
            else:
                plots[k] = v

        for key, val in plots.items():
            fig, ax = plt.subplots()
            pcm = ax.imshow(val, cmap="bwr")
            fig.colorbar(pcm, ax=ax)
            ax.set_title(key)

        plt.show()

    def compute_potentials(self):

        rmc = cp.asarray(self.rmc)
        rp = cp.asarray(self.rp)
        dSp = cp.asarray(self.dSp)
        Fpn = cp.array(self.fm)

        print("interaction displacement vector (2 x M x M)")
        Rm = cp.linalg.norm(rmc[..., None, None, :] - rp, axis=-1)

        print("Greens function (2 x M x n) ...")
        G = cp.exp(-1j * self.k * Rm) / Rm
        G[~cp.isfinite(G)] = 0

        # broadcast integration points

        GmdSp, dFpn = cp.broadcast_arrays(
            cp.asarray(G * dSp),
            cp.asarray(self.dfm[..., None]),
        )

        # reshape along integration points
        GmdSp = GmdSp.reshape(*GmdSp.shape[:2], -1)
        dFpn = dFpn.reshape(*dFpn.shape[:2], -1).transpose(0, 2, 1)
        Fpn = Fpn.reshape(*Fpn.shape[:2], -1, 3).transpose(0, 2, 1, 3)

        print("computing magnetic vector potential ...")
        Avec = [
            [GmdSp[i] @ csr_matrix(Fpn[i, ..., j]) for i in range(2)] for j in range(3)
        ]

        print("computing electric potential ...")
        phi = [GmdSp[i] @ csr_matrix(dFpn[i]) for i in range(2)]

        print("summing ...")
        self.Avec = mu_0 / (4 * np.pi) * cp.stack(cp.array(Avec), axis=-1).get()
        self.phi = -1 / (4 * np.pi * 1j * self.omega * epsilon_0) * cp.array(phi).get()

        print("saving data ...")
        self.data = {
            # "$(r' \\in T_m^\\pm)$": np.sum(ind, axis=0),
            # "$R_m^+$": Rm[0],
            # "$R_m^-$": Rm[1],
            # "R_m==0": np.any(Rm == 0, axis=0),
            # "$G_m(r')^+$": G[0],
            # "$G_m(r')^-$": G[1],
            "$A^+$": np.linalg.norm(self.Avec[0], axis=-1),
            "$A^-$": np.linalg.norm(self.Avec[1], axis=-1),
            "$\\phi^+$": self.phi[0],
            "$\\phi^-$": self.phi[1],
            # "Z": Z,
        }


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * np.linalg.norm(np.cross(u, v, axis=-1), axis=-1)


def demo():
    """run EFIE MoM RWG solver demo problem"""
    from glimmer import Gaussian, Mirror

    src = Gaussian(w0=10 * milli, lam=c / (95 * giga), num_lam=3, num_waist=3)
    src.rotate_z(5)
    src.rotate_y(3)

    zR = np.pi * np.array(src.w0) ** 2 / src.lam

    s = zR / 4
    cs = np.cos(np.pi / 4)
    w = src.w0 * (1 + (s / zR) ** 2) ** 0.5
    L = src.num_waist * w

    options = {
        "L": (L, L / cs),
        "dL": src.lam / src.num_lam,
        "f": (s * cs, s / cs),
    }

    m1 = Mirror(**options)
    m1.rotate_x(45)
    m1.rotate_z(3)
    m1.translate([0, 0, s])

    m2 = Mirror(**options)
    # m2.rotate_x(-45)
    m2.rotate_x(-100)
    # m2.rotate_z(3)
    m2.translate([0, 2 * s, s])

    # src.radiate(m1)
    grid = src + m1  # + m2

    mom = MoM(grid, k=src.k)
    mom.plot_excitation()
    # mom.plot_rwg_basis()
    mom.solve()

    # mom.plot_mesh(scalars="Er")
    mom.plot_currents()

    # mom.show_charts()


if __name__ == "__main__":

    demo()
