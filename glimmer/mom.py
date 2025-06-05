import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

from pyvista import Plotter, PolyData, UnstructuredGrid


eta = 377
mu = 4e-7 * np.pi
eps = 8.854187817e-12


def area(u, v):
    """compute area of triangle defined by vectors u and v"""
    return 0.5*np.linalg.norm(np.cross(u, v, axis=-1), axis=-1)


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

        # edge centers (M x 3)
        self.emc = np.mean(self.rm[0, :2], axis=0)

        # rwg vertex fields ( 2 x 3 x M x 3)
        Em = (self['Er'] + 1j*self['Ei'])[self.con_rwg]

        # rwg centroid fields (2 x M x 3)
        Emc = np.mean(Em, axis=1)

        # areas of faces (N)
        A = area(self.r[1] - self.r[0], self.r[2] - self.r[0])

        # area of rwg faces (2 x M)
        Am = area(self.rm[:, 0] - self.rm[:, 2], self.rm[:, 1] - self.rm[:, 2])

        # rwg edge lengths (M) (v1 to v2)
        lm = np.linalg.norm(self.rm[0, 0] - self.rm[0, 1], axis=-1)

        # match con to rwg connectivity (2 x M x N)
        ind = np.all(
            np.sort(self.con[:, None, :], axis=0) == np.sort(self.con_rwg[..., None], axis=1), axis=1)

        # define kronecker delta function
        I = np.array([1, -1])[:, None, None]

        # compute displacement from face center to free rwg vertex
        rho = (self.rc - self.rm[:, 2][..., None, :])*I[..., None]

        # rwg basis functions (2 x M x N x 3)
        fm = (lm/Am/2)[..., None, None]*rho*ind[..., None]

        # rwg divergence (2 x M x N)
        dfm = (lm/Am)[..., None]*ind*I

        # interaction displacement vector (2 x M x M)
        Rm = np.linalg.norm(self.rmc[..., None, :] - self.rc,  axis=-1)

        # Greens function (2 x M x n)
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.exp(-1j*self.k*Rm)/(4*np.pi*Rm)
            G[ind] = 0

        # angular frequency
        omega = self.k/np.sqrt(mu*eps)

        # prepare moments
        fn = fm[:, None, ...]
        dfn = dfm[:, None, :]
        Gm = G[..., None, :]
        dS = A

        # compute potentials
        Avec = mu*np.nansum(fn*(Gm*dS)[..., None], axis=-2)
        phi = 1j/(omega*eps)*np.nansum(dfn*Gm*dS, axis=-1)

        # centroid basis displacement rwg centroid to free vertex
        self.rhomc = (self.rmc - self.rm[:, 2])*I

        # excitation vector
        V = lm*np.sum(Emc*self.rhomc, axis=(0, -1))/2

        # form impedance matrix
        Z = lm*(1j*omega*np.sum(Avec *
                                self.rhomc[..., None, :], axis=(0, -1)) + (phi[1] - phi[0]))

        # Solve for currents Matrix
        I_n = np.linalg.solve(Z, V)

        # solve for currents
        J = np.sum(I_n[:, None, None]*fm, axis=(0, 1))

        self.cell_data['Jr'] = J.real
        self.cell_data['Ji'] = J.imag
        self.cell_data['J0'] = np.linalg.norm(J, axis=-1)

        # store 2D data
        self.data = {
            "$(r' \in T_m^+) - (r' \in T_m^-)$": ind[0].astype(int) - ind[1].astype(int),
            '$R_m^\pm$': np.sum(Rm, axis=0),
            "$G_m(r')$": np.sum(G, 0),
            '$A$': np.sum(np.linalg.norm(Avec, axis=-1), axis=0),
            '$\phi$': np.sum(phi, axis=0),
            'Z': Z,
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
            sorted_edges_raveled, axis=-1, return_inverse=True, return_counts=True)

        # (2 x M) define rwg basis of unique internal edges
        unique_internal_edges = unique_edges[:, counts == 2]

        corners = []
        for edge in unique_internal_edges.T:

            # map edge to forward and reverse face
            fp = np.all(edge[:, None] == edges, axis=1)
            fn = np.all(edge[:, None] == edges[:, ::-1], axis=1)

            # map edge to edges
            Tp = np.unique(edges[..., np.any(fp, axis=0)])
            Tn = np.unique(edges[..., np.any(fn, axis=0)])

            # find free vertex
            vp = np.setdiff1d(Tp, edge)[0]
            vn = np.setdiff1d(Tn, edge)[0]

            # find non-edge vertex
            corners.append([vp, vn])

        v1, v2 = unique_internal_edges
        vp, vn = np.stack(corners, axis=-1)

        # rwg connectivity (2 x 3 x M)
        self.con_rwg = np.stack([[v1, v2, vp], [v2, v1, vn]])

    def setup_face_connectivity(self):

        # extract surface from unstructured grid
        surf = self.pec.extract_surface()

        # extract surface from unstructured grid
        super().__init__(surf)

        # convert quad mesh to triangle mesh
        self.triangulate(inplace=True)

        # (3 x N) point connectivity for triangulation (v1, v2, v3)
        self.con = self.faces.reshape(4, -1, order='F')[1:]

    def plot_mesh(self):

        rp, rn = self.rm[:, 2]

        rhomcp, rhomcn = self.rhomc

        plotter = Plotter()

        plotter.add_mesh(self, scalars='J0', show_edges=True)
        plotter.add_point_labels(self.points, range(self.points.shape[0]))
        plotter.add_arrows(rp, rhomcp, color='red')
        plotter.add_arrows(rn - rhomcn, rhomcn, color='blue')

        labels = [[str(x) for x in con.T] for con in self.con_rwg]

        plotter.add_point_labels(rp + rhomcp/2, labels[0], text_color='red')
        plotter.add_point_labels(rn - rhomcn/2, labels[1], text_color='blue')

        plotter.enable_parallel_projection()
        plotter.show_grid()

        plotter.show()

    def show_charts(self):

        plots = {}
        for k, v in self.data.items():

            if np.iscomplexobj(v):
                plots[f'Re({k})'] = np.real(v)
                plots[f'Im({k})'] = np.imag(v)
            else:
                plots[k] = v

        for key, val in plots.items():
            fig, ax = plt.subplots()
            pcm = ax.imshow(val, cmap='bwr')
            fig.colorbar(pcm, ax=ax)
            ax.set_title(key)

        plt.show()
