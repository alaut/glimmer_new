from cupyx.scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

from cupyx.scipy.sparse.linalg import gmres

import cupy as cp

from pyvista import Plotter, PolyData, UnstructuredGrid, Arrow

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

        # extract surface from unstructured grid
        surf = self.pec.extract_surface()

        # extract surface from unstructured grid
        super().__init__(surf)

        # convert quad mesh to triangle mesh
        self.triangulate(inplace=True)

        # (3 x N) point connectivity for triangulation (v1, v2, v3)
        self.con = self.faces.reshape(-1, 4)[:, 1:]

        # setup rwg basis (v1, v2, vp), (v1, v2, vn)
        edges, vp, vn = setup_rwg_connectivity(self.con)

        # define rwg basis connectivity
        self.con_m = np.stack([
            np.stack([edges[:, 0], edges[:, 1], vp], axis=-1),
            np.stack([edges[:, 1], edges[:, 0], vn], axis=-1)])

        # face vertices (N x 3 x 3)
        r = self.points[self.con]

        # rwg vertices (2 x M x 3 x 3)
        rm = self.points[self.con_m]

        # rwg edge length
        lm = np.linalg.norm(rm[0, :, 0] - rm[0, :, 1], axis=-1)

        # rwg areas
        Am = area(rm[:, :, 0] - rm[:, :, -1], rm[:, :, 1] - rm[:, :, -1])

        # face centroids (N x 3)
        rc = np.mean(r, axis=-2)

        # rwg centroids (2 x M x 3)
        rmc = np.mean(rm, axis=-2)

        # rwg vertex fields (2 x 3 x M x 3)
        Em = (self["Er"] + 1j * self["Ei"])[self.con_m]

        # rwg centroid fields (2 x M x 3)
        Emc = np.mean(Em, axis=-2)

        # areas of faces (N)
        dSp = area(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0])

        # define plus/minus modifier (used to correct rho vector orientation)
        pm = np.array([1, -1])[:, None, None]

        # match con to rwg connectivity (2 x M x N)
        ind = np.all(np.sort(self.con) == np.sort(
            self.con_m)[..., None, :], axis=-1)

        pm = np.array([1, -1])[:, None, None]

        # define source points r' at centroids
        rp = rc

        # rwg basis functions; compute displacement from face center to free rwg vertex (2 x M x N x 3)
        rho = pm[..., None]*(rp - rm[..., [-1], :])
        fm = 0.5 * (lm / Am)[..., None, None] * rho*ind[..., None]

        # rwg divergence (2 x M x N)
        dfm = pm*(lm/Am)[..., None]*ind

        # interaction displacement vector (2 x M x M)
        Rm = np.linalg.norm(rmc[..., None, :] - rp, axis=-1)

        # Greens function (2 x M x n)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.exp(-1j * self.k * Rm) / Rm
            G[np.isnan(G)] = 0

        # angular frequency
        omega = self.k / np.sqrt(mu * eps)

        Avec, phi = compute_potentials(fm, dfm, G * dSp, omega)

        # centroid basis displacement rwg centroid to free vertex
        rhomc = pm * (rmc - rm[:, :, -1])

        # excitation vector
        V = lm * np.sum(Emc * rhomc / 2, axis=(0, -1))

        # form impedance matrix
        Z = lm * np.sum(1j * omega * np.sum(Avec * rhomc[..., None, :]/2,   axis=-1) + phi[0] - phi[1],
                        axis=0)

        # solve basis currents
        I = gmres(cp.asarray(Z), cp.asarray(V))[0].get()

        # solve for unknown surface currents
        self.J = np.sum(I[:, None, None] * fm, axis=(0, 1))

        J = self.J
        J02 = np.linalg.norm(J, axis=-1) ** 2

        self.cell_data["Jr"] = J.real
        self.cell_data["Ji"] = J.imag
        self.cell_data["|J|^2"] = J02
        self.cell_data["|J|^2 (dB)"] = 10 * np.log(J02 / J02.max())

        # store 2D data
        self.data = {
            "$(r' \\in T_m^+) - (r' \\in T_m^-)$": ind[0].astype(int) - ind[1].astype(int),
            "$R_m^\\pm$": np.sum(Rm, axis=0),
            "$G_m(r')$": np.sum(G, 0),
            "$A$": np.sum(np.linalg.norm(Avec, axis=-1), axis=0),
            "$\\phi$": np.sum(phi, axis=0),
            "Z": Z,
        }

        self.rm = rm
        self.rhomc = rhomc
        self.rp = rp
        self.rc = rc
        self.rmc = rmc

    def plot_mesh(self, faces=False, basis_functions=False,  lines=False):

        plotter = Plotter()
        plotter.add_mesh(self, scalars="|J|^2", show_edges=True)

        rp, rn = self.rm[:, :, -1]

        rhomcp, rhomcn = self.rhomc

        if faces:
            plotter.add_point_labels(self.points, range(self.points.shape[0]))
            plotter.add_point_labels(self.rc, [str(con) for con in self.con])

        if basis_functions:
            plotter.add_arrows(rp, rhomcp, color="red")
            plotter.add_arrows(rn - rhomcn, rhomcn, color="blue")

            plotter.add_point_labels(
                rp + rhomcp / 2, [str(x) for x in self.con_m[0]], text_color="red")
            plotter.add_point_labels(
                rn - rhomcn / 2, [str(x) for x in self.con_m[1]], text_color="blue")

        plotter.add_points(self.rp, render_points_as_spheres=True)
        plotter.add_points(self.rc, render_points_as_spheres=True)

        if lines:
            i, j, k = np.indices(self.Rm.shape)
            lines = plotter.add_lines(
                lines=np.stack([self.rp[k], self.rmc[i, j]]).reshape(-1, 3, order='F'), width=0.1,
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


def setup_rwg_connectivity(con):
    """compute rwg basis of v1, v2, vp and vn"""

    # construct edges from face connectivity (face x edge x point)
    edges = con[:, [[0, 1], [1, 2], [2, 0]]]

    # sort edges by points
    sorted_edges = np.sort(edges, axis=-1)

    # count uniqueness among flattened edges
    unique_edges, counts = np.unique(
        sorted_edges.reshape(-1, 2), axis=0, return_counts=True)

    # (2 x M) define rwg basis of unique internal edges
    internal_edges = unique_edges[counts == 2]

    # find free rwg vertices
    free_vert_p = [get_free_vert(edge, edges) for edge in internal_edges]
    free_vert_n = [get_free_vert(edge[::-1], edges) for edge in internal_edges]

    return internal_edges, np.array(free_vert_p), np.array(free_vert_n)


def get_free_vert(edge, face_edges):
    """find free vertex given edge in face edges"""

    # match points to edges
    ind = np.all(edge == face_edges, axis=-1)

    # match edges to face
    face = np.any(ind, axis=-1)

    # get face to points
    points = np.unique(face_edges[face])

    # find free vertex
    vertex = np.setdiff1d(points, edge)

    return int(vertex[0])


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

    src = Gaussian(w0=10, lam=3, num_lam=4, num_waist=2)
    # src.rotate_z(5)
    src.rotate_y(3)

    zR = np.pi * np.array(src.w0)**2 / src.lam

    s = zR / 4
    c = np.cos(np.pi / 4)
    w = src.w0 * (1 + (s / zR) ** 2) ** 0.5

    options = {
        "L": (src.num_waist * w, src.num_waist * w / c),
        "dL": src.lam / src.num_lam,
        "f": (s * c, s / c),
        # "f": None,
    }

    m1 = Mirror(**options)
    m1.rotate_x(45)

    m1.translate([0, 0, s])
    m1.round()
    src.radiate(m1)

    m2 = Mirror(**options)
    m2.rotate_x(-45)
    m2.translate([0, 2*s, s])
    m2.round()
    # src.radiate(m2)

    mom = MoM(
        pec=m1,
        # pec=m1+m2,
        k=src.k)

    mom.plot_mesh()

    mom.show_charts()
