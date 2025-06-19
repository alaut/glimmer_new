import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix

from cupyx.scipy.sparse.linalg import gmres


from pyvista import Plotter, PolyData, UnstructuredGrid
from glimmer import eps, mu, eta


@dataclass
class MoM(PolyData):

    k: float
    pec: UnstructuredGrid

    # s: float = 0.5
    s: float = None

    order: str = "C"

    def __post_init__(self):

        # extract surface from unstructured grid
        surf = self.pec.extract_surface()

        # extract surface from unstructured grid
        super().__init__(surf)

        # convert quad mesh to triangle mesh
        self.triangulate(inplace=True)
        self.clean()

        # (3 x N) point connectivity for triangulation (v1, v2, v3)
        self.con = self.faces.reshape(-1, 4)[:, 1:]

        N = self.con.shape[0]

        # setup rwg basis (v1, v2, vp), (v1, v2, vn)
        edges, vp, vn = setup_rwg_connectivity(self.con)

        # define rwg basis connectivity
        self.con_m = np.stack(
            [
                np.stack([edges[:, 0], edges[:, 1], vp], axis=-1),
                np.stack([edges[:, 1], edges[:, 0], vn], axis=-1),
            ]
        )

        # face vertices (N x 3 x 3)
        r = self.points[self.con]

        # rwg vertices (2 x M x 3 x 3)
        rm = self.points[self.con_m]

        M = self.con_m.shape[1]

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
        ind = pm * np.all(np.sort(self.con) == np.sort(self.con_m)[:, :, None], axis=-1)

        # define source points r' at centroids (N, n, 3)
        rp = r + self.s * (rc[:, None] - r) if self.s else rc[:, None, :]

        omega = self.k / np.sqrt(eps * mu)

        n = rp.shape[1]
        ind = np.stack(n * [ind], axis=-1).reshape((2, M, -1), order=self.order)
        dSp = np.stack(n * [dSp / n], axis=-1).reshape(-1, order=self.order)
        rp = rp.reshape((-1, 3), order=self.order)

        # rwg basis functions; compute displacement from face center to free rwg vertex (2 x M x N x 3)
        rho = rp - rm[..., -1, :][..., None, :]
        fm = (ind * (lm / Am)[..., None])[..., None] * rho

        # rwg divergence (2 x M x N)
        dfm = ind * (lm / Am)[..., None]

        # interaction displacement vector (2 x M x M)
        Rm = np.linalg.norm(rmc[..., None, :] - rp, axis=-1)

        # Greens function (2 x M x n)
        G = np.exp(-1j * self.k * Rm) / Rm
        G[np.nonzero(Rm == 0)] = 0

        Avec, phi = compute_potentials(fm, dfm, G * dSp, omega)

        # centroid basis displacement rwg centroid to free vertex
        rhomc = pm * (rmc - rm[:, :, -1])

        # excitation vector
        V = lm * np.sum(Emc * rhomc / 2, axis=(0, -1))

        # form impedance matrix
        Z = lm * (
            1j * omega * np.sum(Avec * rhomc[:, :, None, :] / 2, axis=(0, -1))
            + (phi[0] - phi[1])
        )

        # solve basis currents
        print("solving ...")
        I, info = gmres(cp.asarray(Z), cp.asarray(V))
        I = I.get()

        # solve for unknown surface currents
        self.Jp = np.sum(I[:, None, None] * fm, axis=(0, 1))

        # average about source points
        J = np.mean(np.reshape(self.Jp, (N, n, 3), order=self.order), axis=1)

        # Compute magnitude squared for plotting
        J02 = np.linalg.norm(J, axis=-1)

        self.cell_data["Jr"] = J.real
        self.cell_data["Ji"] = J.imag
        self.cell_data["|J|^2"] = J02
        self.cell_data["|J|^2 (dB)"] = 10 * np.log(J02 / J02.max())

        # store 2D data
        self.data = {
            "$(r' \\in T_m^\\pm)$": np.sum(ind, axis=0),
            "$R_m^+$": Rm[0],
            "$R_m^-$": Rm[1],
            "R_m==0": np.any(Rm == 0, axis=0),
            "$G_m(r')^+$": G[0],
            "$G_m(r')^-$": G[1],
            "$A^+$": np.linalg.norm(Avec[0], axis=-1),
            "$A^-$": np.linalg.norm(Avec[1], axis=-1),
            "$\\phi^+$": phi[0],
            "$\\phi^-$": phi[1],
            "Z": Z,
        }

        self.rm = rm
        self.rhomc = rhomc
        self.rp = rp
        self.rc = rc
        self.rmc = rmc

    def plot_mesh(self, faces=False, basis_functions=False, lines=False):

        plotter = Plotter()
        plotter.add_mesh(
            self,
            #  scalars="Jr",
            scalars="|J|^2",
            show_edges=True,
        )

        rp, rn = self.rm[:, :, -1]

        rhomcp, rhomcn = self.rhomc

        if faces:
            plotter.add_point_labels(self.points, range(self.points.shape[0]))
            plotter.add_point_labels(self.rc, [str(con) for con in self.con])

        if basis_functions:
            plotter.add_arrows(rp, rhomcp, color="red")
            plotter.add_arrows(rn - rhomcn, rhomcn, color="blue")

            plotter.add_point_labels(
                rp + rhomcp / 2, [str(x) for x in self.con_m[0]], text_color="red"
            )
            plotter.add_point_labels(
                rn - rhomcn / 2, [str(x) for x in self.con_m[1]], text_color="blue"
            )

        plotter.add_points(self.rp, render_points_as_spheres=True)
        plotter.add_points(self.rc, render_points_as_spheres=True, color="magenta")

        if lines:
            i, j, k = np.indices(self.Rm.shape)
            lines = plotter.add_lines(
                lines=np.stack([self.rp[k], self.rmc[i, j]]).reshape(-1, 3, order="F"),
                width=0.1,
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


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5 * np.linalg.norm(np.cross(u, v, axis=-1), axis=-1)


def setup_rwg_connectivity(con):
    """given trimesh connectivity con (N, 3), construct rwg basis of M unique internal edges"""

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

    return internal_edges, vert_pos, vert_neg


def compute_potentials(fm, dfm, GdSp, omega):

    print("computing potentials ...")

    dF = cp.array(dfm).transpose(0, 2, 1)
    F = cp.array(fm).transpose(0, 2, 1, 3)
    G = cp.array(GdSp)

    I = range(2)
    J = range(3)

    Avec = [[G[i] @ csr_matrix(F[i, ..., j]) for i in I] for j in J]

    phi = [G[i] @ csr_matrix(dF[i]) for i in I]

    Avec = mu / (4 * np.pi) * cp.stack(cp.array(Avec), axis=-1).get()
    phi = -1 / (4 * np.pi * 1j * omega * eps) * cp.array(phi).get()

    return Avec, phi


def demo():
    """run EFIE MoM RWG solver demo problem"""
    from glimmer import Gaussian, Mirror

    src = Gaussian(w0=10e-3, lam=3e8 / 95e9, num_lam=3, num_waist=2)
    src.rotate_z(5)
    src.rotate_y(3)

    zR = np.pi * np.array(src.w0) ** 2 / src.lam

    s = zR / 4
    c = np.cos(np.pi / 4)
    w = src.w0 * (1 + (s / zR) ** 2) ** 0.5

    options = {
        "L": (src.num_waist * w, src.num_waist * w / c),
        "dL": src.lam / src.num_lam,
        "f": (s * c, s / c),
    }

    m1 = Mirror(**options)
    m1.rotate_x(45)
    m1.rotate_z(3)
    m1.translate([0, 0, s])

    src.radiate(m1)

    mom = MoM(pec=m1, k=src.k, s=0.5)

    mom.plot_mesh()

    mom.show_charts()


if __name__ == "__main__":

    demo()
