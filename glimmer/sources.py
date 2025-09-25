import pyvista as pv
import numpy as np
from dataclasses import dataclass
from scipy.constants import mu_0, epsilon_0
from scipy.special import jnp_zeros, jv, jvp

eta = np.sqrt(mu_0 / epsilon_0)

from . import set_field
@dataclass
class TransverseElectric(pv.StructuredGrid):

    lam: float

    m: int
    n: int

    a: float = None
    omf: float = 1.25

    num_lam: float = 12

    rmin: float = 0

    nr: int = None
    nt: int = None

    def __post_init__(self):

        k0 = 2 * np.pi / self.lam

        nu = jnp_zeros(self.m, self.n)[-1]

        if self.a is None:
            self.a = self.omf * nu / k0
        else:
            self.omf = self.a / (nu / k0)

        dl = self.lam / self.num_lam

        if self.nr is None:
            self.nr = int(np.max([self.n * self.num_lam, (self.a - self.rmin) / dl]))
        if self.nt is None:
            self.nt = int(np.max([self.m * self.num_lam, 2 * np.pi * self.a / dl]))

        R, T, Z = np.meshgrid(
            np.linspace(self.rmin, self.a, self.nr),
            np.linspace(0, 2 * np.pi, self.nt),
            0,
            indexing="ij",
        )

        super().__init__(R * np.cos(T), R * np.sin(T), Z)

        x = self.points[..., 0]
        y = self.points[..., 1]

        r = np.sqrt(x**2 + y**2)
        t = np.atan2(y, x)

        kr = nu / self.a
        kz = np.sqrt(k0**2 - kr**2)

        J = jv(self.m, kr * r)
        Jp = jvp(self.m, kr * r)

        C = np.cos(self.m * t)
        S = np.sin(self.m * t)

        Hz = J * C
        Hr = -1j * kz / kr * Jp * C
        with np.errstate(invalid="ignore", divide="ignore"):
            Ht = 1j * self.m * kz / r / kr**2 * J * S
            Ht[r == 0] = 0

        Z = k0 / kz * eta

        Er = Z * Ht
        Et = -Z * Hr

        Ex = np.cos(t) * Er - np.sin(t) * Et
        Ey = np.sin(t) * Er + np.cos(t) * Et
        Ez = np.zeros_like(r)

        Hx = np.cos(t) * Hr - np.sin(t) * Ht
        Hy = np.sin(t) * Hr + np.cos(t) * Ht

        E = np.stack([Ex, Ey, Ez], axis=-1)
        H = np.stack([Hx, Hy, Hz], axis=-1)

        set_field(self, E, "E")
        set_field(self, H, "H")


# from glimmer.mom import set_fields


if __name__ == "__main__":

    te = TransverseElectric(lam=300 / 110, m=22, n=6, omf=1.06, rmin=8)
    te.plot()
