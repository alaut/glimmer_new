import pyvista as pv
import numpy as np
from dataclasses import dataclass
from scipy.constants import mu_0, epsilon_0
from scipy.special import jnp_zeros, jv, jvp

eta = np.sqrt(mu_0 / epsilon_0)


@dataclass
class TransverseElectric(pv.StructuredGrid):

    lam: float
    m: int
    n: int

    a: float = None
    omf: float = 1.25

    nr: int = 2 * 2
    nt: int = 2 * 6 * 2

    def __post_init__(self):

        self.k0 = 2 * np.pi / self.lam

        nu = jnp_zeros(self.m, self.n)[-1]

        if self.a is None:
            self.a = self.omf * nu / self.k0
        else:
            self.omf = self.a / (nu / self.k0)

        kr = nu / self.a
        kz = np.sqrt(self.k0**2 - kr**2)

        # caustic radius
        ac = self.a * self.m / nu

        R, T, Z = np.broadcast_arrays(
            np.linspace(self.a, ac / 2, self.nr, endpoint=False)[..., None],
            np.linspace(0, 2 * np.pi, self.nt),
            0,
        )
        X = R * np.cos(T)
        Y = R * np.sin(T)

        super().__init__(X, Y, Z)

        x, y, z = np.moveaxis(self.points, -1, 0)

        r = np.sqrt(x**2 + y**2)
        t = np.atan2(y, x)

        J = jv(self.m, kr * r)
        Jp = jvp(self.m, kr * r)

        C = np.cos(self.m * t)
        S = np.sin(self.m * t)

        Hz = J * C
        Hr = -1j * kz / kr * Jp * C
        Ht = 1j * self.m * kz / r / kr**2 * J * S

        Z = self.k0 / kz * eta

        Er = Z * Ht
        Et = -Z * Hr

        Ex = np.cos(t) * Er - np.sin(t) * Et
        Ey = np.sin(t) * Er + np.cos(t) * Et
        Ez = np.zeros_like(z)

        Hx = np.cos(t) * Hr - np.sin(t) * Ht
        Hy = np.sin(t) * Hr + np.cos(t) * Ht

        E = np.stack([Ex, Ey, Ez], axis=-1)
        H = np.stack([Hx, Hy, Hz], axis=-1)

        self["Er"] = E.real
        self["Ei"] = E.imag
        self["Hr"] = H.real
        self["Hi"] = H.imag

        self["|E|^2"] = np.linalg.norm(E, axis=-1) ** 2
        self["|H|^2"] = np.linalg.norm(H, axis=-1) ** 2
