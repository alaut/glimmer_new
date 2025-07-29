import quadpy

from dataclasses import dataclass

import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres


import pyvista as pv

from scipy.constants import epsilon_0, mu_0, c, milli, giga


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
        self.setup_connectivity()

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

    def setup_connectivity(self):
        """generate RWG (Rao, Wilton, Glisson) connectivity from trimesh face connectivity"""

        self.con = self.faces.reshape(-1, 4)[:, 1:]

        # construct edges from face connectivity (face x edge x point) (N, 3, 2)
        edges = self.con[:, [[0, 1], [1, 2], [2, 0]]]

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

        self.con_m = np.stack(
            [
                np.stack(
                    [internal_edges[:, 0], internal_edges[:, 1], vert_pos], axis=-1
                ),
                np.stack(
                    [internal_edges[:, 1], internal_edges[:, 0], vert_neg], axis=-1
                ),
            ]
        )

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

        # fix
        print("forming impedance matrix ...")
        self.Z = self.lm * (
            1j
            * self.omega
            * np.sum(self.Avec * self.rhomc[:, :, None, :] / 2, axis=(0, -1))
            - (self.phi[1] - self.phi[0])
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

        factor = milli / np.max(np.linalg.norm(self["Jr"], axis=-1))

        plotter = pv.Plotter()
        plotter.add_mesh(self, scalars=scalars, show_edges=True, cmap="jet")
        plotter.add_mesh(self.glyph("Jr", factor=factor), color="blue")
        plotter.add_mesh(self.glyph("Ji", factor=factor), color="red")

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

    def radiate(self, probe: pv.StructuredGrid):
        """radiate current sources to probe"""

        print("radiating EFIE ...")

        dSp = cp.asarray(self.compute_cell_sizes()["Area"])[..., None]

        J = cp.asarray(self.cell_data["Jr"] + 1j * self.cell_data["Ji"])

        r2 = cp.asarray(probe.points[..., None, :])
        r1 = cp.asarray(self.cell_centers().points)

        r = r2 - r1

        R = cp.linalg.norm(r, axis=-1, keepdims=True)

        G = cp.exp(-1j * self.k * R) / (4 * cp.pi * R)

        # re-derive using green't function with positive exponent
        dG = r * (G / R**2 + 1j * self.k * G / R)

        A = mu_0 * cp.sum(J * G * dSp, axis=-2)

        divJr = self.compute_derivative("Jr", divergence=True)["divergence"]
        divJi = self.compute_derivative("Jr", divergence=True)["divergence"]
        divJ = cp.asarray(divJr + 1j * divJi)[..., None]

        grad_phi = cp.sum(1j * divJ / self.omega * dG * dSp, axis=-2)

        probe.point_data["Ar"] = A.get().real
        probe.point_data["Ai"] = A.get().imag

        Es = -1j * self.omega * A - grad_phi

        probe.point_data["Er"] = Es.get().real
        probe.point_data["Ei"] = Es.get().imag
        probe.point_data["|E|^2"] = np.linalg.norm(Es.get(), axis=-1) ** 2


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * np.linalg.norm(np.cross(u, v, axis=-1), axis=-1)


def poly2grid(poly, d, x=None, y=None, z=None, l=0):

    xmin, xmax, ymin, ymax, zmin, zmax = poly.bounds

    nx = int((xmax - xmin + 2 * l) / d) + 1
    ny = int((ymax - ymin + 2 * l) / d) + 1
    nz = int((zmax - zmin + 2 * l) / d) + 1

    x = np.linspace(xmin - l, xmax + l, nx) if x is None else float(x)
    y = np.linspace(ymin - l, ymax + l, ny) if y is None else float(y)
    z = np.linspace(zmin - l, zmax + l, nz) if z is None else float(z)

    x, y, z = np.meshgrid(x, y, z)

    return pv.StructuredGrid(np.squeeze(x), np.squeeze(y), np.squeeze(z))
