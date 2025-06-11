from cupyx.scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

from cupyx.scipy.sparse.linalg import gmres

import cupy as cp

from pyvista import Plotter, PolyData, UnstructuredGrid

eta = 377
mu = 4e-7 * np.pi
eps = 8.854187817e-12


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * np.linalg.norm(np.cross(u, v, axis=-1), axis=-1)


@dataclass
class MoM(PolyData):

    k: float
    pec: UnstructuredGrid

    def __post_init__(self):

        # setup face (v1, v2, v3)
        self.setup_face_connectivity()

        # setup rwg basis (v1, v2, vp), (v1, v2, vn)
        self.setup_rwg_connectivity()

        # face vertices (3 x N x 3)
        self.r = self.points[self.con]

        # rwg vertices (2 x 3 x M x 3)
        self.rm = self.points[self.con_rwg]

        # face centroids (N x 3)
        self.rc = np.mean(self.r, axis=0)

        # rwg centroids (2 x M x 3)
        self.rmc = np.mean(self.rm, axis=1)

        # rwg vertex fields (2 x 3 x M x 3)
        Em = (self["Er"] + 1j * self["Ei"])[self.con_rwg]

        # rwg centroid fields (2 x M x 3)
        Emc = np.mean(Em, axis=1)

        # areas of faces (N)
        A = area(self.r[0] - self.r[-1], self.r[1] - self.r[-1])

        # face vertex to centroid midpoints
        self.rp = self.rc
        self.rp = (self.rc + self.r[0]) / 2
        self.rp = 0.5 * (self.rc + self.r).reshape(-1, 3, order="F")

        # kite area of integrated component
        dSp = A
        dSp = np.stack([A / 3, A / 3, A / 3]).reshape(-1, order="F")

        # area of rwg faces (2 x M)
        Am = area(self.rm[:, 0] - self.rm[:, -1],
                  self.rm[:, 1] - self.rm[:, -1])

        # rwg edge lengths (M) (v1 to v2)
        lm = np.linalg.norm(self.rm[0, 0] - self.rm[0, 1], axis=-1)

        # define plu/minus modifier (used to correct rho vector orientation)
        pm = np.array([1, -1])[:, None, None]

        # match con to rwg connectivity (2 x M x N)
        ind = np.all(
            np.sort(self.con[:, None, :], axis=0)
            == np.sort(self.con_rwg[..., None], axis=1),
            axis=1,
        )

        ind = np.stack([ind, ind, ind], axis=-2)
        ind = ind.reshape(ind.shape[0], ind.shape[1], -1, order="F")

        # rwg basis functions; compute displacement from face center to free rwg vertex (2 x M x N x 3)
        rho = pm[..., None] * (self.rp - self.rm[:, -1][..., None, :])
        fm = 0.5 * (lm / Am)[..., None, None] * rho * ind[..., None]

        # rwg divergence (2 x M x N)
        dfm = pm * (lm / Am)[..., None] * ind

        # interaction displacement vector (2 x M x M)
        Rm = np.linalg.norm(self.rmc[..., None, :] - self.rp, axis=-1)

        # Greens function (2 x M x n)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.exp(-1j * self.k * Rm) / Rm

        # angular frequency
        omega = self.k / np.sqrt(mu * eps)

        Avec, phi = compute_potentials(fm, dfm, G * dSp, omega)

        # centroid basis displacement rwg centroid to free vertex
        self.rhomc = pm * (self.rmc - self.rm[:, -1])

        # excitation vector
        V = lm * np.sum(Emc * self.rhomc / 2, axis=(0, -1))

        # form impedance matrix
        Z = lm * (
            1j * omega *
            np.sum(Avec * self.rhomc[..., None, :], axis=(0, -1)) / 2
            + phi[0]
            - phi[1]
        )

        # solve basis currents
        I = gmres(cp.asarray(Z), cp.asarray(V))[0].get()

        # solve for unknown surface currents
        self.J = np.sum(I[:, None, None] * fm, axis=(0, 1))
        J = (self.J[::3] + self.J[1::3] + self.J[2::3]) / 3
        J02 = np.linalg.norm(J, axis=-1) ** 2

        self.cell_data["Jr"] = J.real
        self.cell_data["Ji"] = J.imag
        self.cell_data["|J|^2"] = J02
        self.cell_data["|J|^2 (dB)"] = 10 * np.log(J02 / J02.max())

        # store 2D data
        self.data = {
            "$(r' \\in T_m^+) - (r' \\in T_m^-)$": ind[0].astype(int)
            - ind[1].astype(int),
            "$R_m^\\pm$": np.sum(Rm, axis=0),
            "$G_m(r')$": np.sum(G, 0),
            "$A$": np.sum(np.linalg.norm(Avec, axis=-1), axis=0),
            "$\\phi$": np.sum(phi, axis=0),
            "Z": Z,
        }

    def setup_rwg_connectivity(self):
        """compute rwg basis of v1, v2, vp and vn"""

        # construct edges from face connectivity
        edges = self.con[[[0, 1], [1, 2], [2, 0]]]

        # sort edge vertices
        sorted_edges = np.sort(edges, axis=1)

        # flatten edges
        sorted_edges_raveled = sorted_edges.transpose(1, 0, 2).reshape(2, -1)

        # count uniqueness
        unique_edges, inverse_indices, counts = np.unique(
            sorted_edges_raveled, axis=-1, return_inverse=True, return_counts=True
        )

        # (2 x M) define rwg basis of unique internal edges
        internal_edges = unique_edges[:, counts == 2]

        def vert(edge):
            """find free vertex given edge in edges"""

            # map edge to face
            face = np.any(np.all(edge[:, None] == edges, axis=1), axis=0)

            # get face to points
            points = np.unique(edges[..., face])

            # find free vertex
            vertex = np.setdiff1d(points, edge)[0]

            return vertex

        # find free rwg vertices
        free_vertices = [[vert(edge), vert(edge[::-1])]
                         for edge in internal_edges.T]

        # define rwg edges
        v1, v2 = internal_edges
        vp, vn = np.stack(free_vertices, axis=-1)

        # rwg connectivity (2 x 3 x M)
        self.con_rwg = np.stack([[v1, v2, vp], [v2, v1, vn]])

        self.M = self.con_rwg.shape[-1]
        print(f"M: {self.M} edges")

    def setup_face_connectivity(self):

        # extract surface from unstructured grid
        surf = self.pec.extract_surface()

        # extract surface from unstructured grid
        super().__init__(surf)

        # convert quad mesh to triangle mesh
        self.triangulate(inplace=True)

        # (3 x N) point connectivity for triangulation (v1, v2, v3)
        self.con = self.faces.reshape(4, -1, order="F")[1:]

        self.N = self.con.shape[-1]
        print(f"N: {self.N} faces")

    def plot_mesh(self, points=True, vectors=True, text=False):

        rp, rn = self.rm[:, -1]

        rhomcp, rhomcn = self.rhomc

        plotter = Plotter()
        plotter.add_mesh(self, scalars="|J|^2", show_edges=True)

        if points:
            Jp = np.linalg.norm(self.J, axis=-1)
            plotter.add_points(
                self.rp.reshape(-1, 3),
                render_points_as_spheres=True,
                scalars=Jp.reshape(-1),
            )
            plotter.add_points(
                self.rc, render_points_as_spheres=True, color="magenta")

        if vectors:
            plotter.add_arrows(rp, rhomcp, color="red")
            plotter.add_arrows(rn - rhomcn, rhomcn, color="blue")

        if text:
            labels = [[str(x) for x in con.T] for con in self.con_rwg]
            plotter.add_point_labels(
                rp + rhomcp / 2, labels[0], text_color="red")
            plotter.add_point_labels(
                rn - rhomcn / 2, labels[1], text_color="blue")
            plotter.add_point_labels(self.points, range(self.points.shape[0]))

        plotter.enable_parallel_projection()
        plotter.show_grid()

        # plotter.show()

        return plotter

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


def compute_potentials(F, dF, GdSp, omega):

    print("computing potentials ...")

    GdSp = cp.asarray(GdSp)
    F = cp.asarray(F)
    dF = cp.asarray(dF)

    I = range(2)
    J = range(3)

    # compute vector potential (2 x M x M x 3)
    Avec = [[GdSp[i] @ csr_matrix(F[i, ..., j].T) for i in I] for j in J]

    # compute scalar potential (2 x M x M)
    phi = [GdSp[i] @ csr_matrix(dF[i].T) for i in I]

    Avec = mu / (4 * np.pi) * cp.stack(cp.array(Avec), axis=-1)
    phi = -1 / (4 * np.pi * 1j * omega * eps) * cp.array(phi)

    print("finished computing potentials !")

    return Avec.get(), phi.get()


if __name__ == "__main__":

    from glimmer import Gaussian, Mirror

    src = Gaussian(w0=10, lam=3, num_lam=3, num_waist=2)
    src.rotate_z(5)
    src.rotate_y(3)

    s = src.zR / 2
    c = np.cos(np.pi / 4)
    w = src.w0 * (1 + (s / src.zR) ** 2) ** 0.5

    options = {
        "L": (src.num_waist * w, src.num_waist * w / c),
        "dL": src.lam / src.num_lam,
        #  "f": (s * c, s / c)
        "f": None,
    }

    m1 = Mirror(**options)
    m1.rotate_x(45)
    m1.translate([0, 0, s])
    m1.round()

    m2 = Mirror(**options)
    m2.rotate_x(-45)
    m2.translate([0, 2 * s, s])
    m2.round()

    src.radiate(m1)
    src.radiate(m2)

    mom = MoM(
        pec=m1 + m2,
        k=src.k,
    )

    plotter = mom.plot_mesh()
    plotter.add_mesh(src, scalars='|E|^2')
    plotter.show()

    mom.show_charts()
