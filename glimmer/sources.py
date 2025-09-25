from scipy.special import factorial
import pyvista as pv
import numpy as np
from dataclasses import dataclass
from scipy.constants import mu_0, epsilon_0
from scipy.special import jnp_zeros, jv, jvp

eta = np.sqrt(mu_0 / epsilon_0)

# try:
#     # from . import set_field, Grid, add_fields
from . import *
from .tools import integrate_power

# except:
#     from glimmer import *
#     from glimmer.tools import integrate_power


def GaussianBeam(w0, lam, num_lam=3, num_waist=3, P0=1):
    """generate gaussian source object"""

    w0 = np.array([1, 1]) * np.array(w0)
    Lx, Ly = w0 * num_waist

    grid = Grid(xlim=(-Lx / 2, Lx / 2), ylim=(-Ly / 2, Ly / 2), d=lam / num_lam)

    X = grid.points[..., 0] / w0[0]
    Y = grid.points[..., 1] / w0[1]

    I0 = 2 * P0 / (np.pi * w0[0] * w0[1])
    A = np.sqrt(I0 * eta) * np.exp(-(X**2)) * np.exp(-(Y**2))

    add_fields(
        grid,
        E=A[..., None] * np.array([1, 0, 0]),
        H=A[..., None] * np.array([0, 1, 0]) / eta,
    )

    return grid


def TransverseElectric(
    lam: float,
    m: int,
    n: int,
    a: float = None,
    omf: float = 1.25,
    num_lam: float = 12,
    rmin: float = 0,
    nr: int = None,
    nt: int = None,
):

    k0 = 2 * np.pi / lam

    nu = jnp_zeros(m, n)[-1]

    if a is None:
        a = omf * nu / k0
    else:
        omf = a / (nu / k0)

    dl = lam / num_lam

    if nr is None:
        nr = int(np.max([n * num_lam, (a - rmin) / dl]))
    if nt is None:
        nt = int(np.max([m * num_lam, 2 * np.pi * a / dl]))

    R, T, Z = np.meshgrid(
        np.linspace(rmin, a, nr),
        np.linspace(0, 2 * np.pi, nt),
        0,
        indexing="ij",
    )
    grid = pv.StructuredGrid(R * np.cos(T), R * np.sin(T), Z)

    x = grid.points[..., 0]
    y = grid.points[..., 1]

    r = np.sqrt(x**2 + y**2)
    t = np.atan2(y, x)

    kr = nu / a
    kz = np.sqrt(k0**2 - kr**2)

    J = jv(m, kr * r)
    Jp = jvp(m, kr * r)

    C = np.cos(m * t)
    S = np.sin(m * t)

    Hz = J * C
    Hr = -1j * kz / kr * Jp * C
    with np.errstate(invalid="ignore", divide="ignore"):
        Ht = 1j * m * kz / r / kr**2 * J * S
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

    set_field(grid, E, "E")
    set_field(grid, H, "H")

    return grid


def HermiteGaussian(lam=300 / 95, w0=10, m=0, n=0, c=1, num_waist=3, num_lam=6):
    """Goldsmith 1998 eq 2.62"""

    wx, wy = np.array([1, 1]) * np.array(w0)

    m = np.atleast_1d(m)
    n = np.atleast_1d(n)
    c = np.atleast_1d(c)

    Lx = wx * num_waist * np.sqrt(m.max() + 1)
    Ly = wy * num_waist * np.sqrt(n.max() + 1)

    grid = Grid(xlim=(-Lx / 2, Lx / 2), ylim=(-Ly / 2, Ly / 2), d=lam / num_lam)

    mf = factorial(m)
    nf = factorial(n)

    x = grid.points[..., 0]
    y = grid.points[..., 1]

    Hm = hermite(m, np.sqrt(2) * x / wx)
    Hn = hermite(n, np.sqrt(2) * y / wy)

    A0 = Hm * Hn / np.sqrt(np.pi * wx * wy * 2.0 ** (m + n - 1) * mf * nf)

    X = np.exp(-((x / wx) ** 2))[..., None]
    Y = np.exp(-((y / wy) ** 2))[..., None]
    A = np.sum(c * A0 * X * Y, axis=-1)

    # normalizes power
    A = A * 10 * np.sqrt(30 / 8)

    add_fields(
        grid,
        E=A[..., None] * np.array([1, 0, 0]),
        H=A[..., None] * np.array([0, 1, 0]) / eta,
    )

    return grid


def hermite(m, x):
    """Goldsmith 1998 eq. 2.58"""

    y = (m == 0) + 2 * (m == 1) * x[..., None]

    ind = m > 1

    if any(ind):

        h1 = x[..., None] * hermite(m[ind] - 1, x)
        h2 = (m[ind] - 1) * hermite(m[ind] - 2, x)

        y[..., ind] = 2 * (h1 - h2)

    return y


# def factorial(n):
#     """returns factorial"""

#     y = (n == 0).astype(float) + (n == 1).astype(float)

#     ind = n > 1

#     if any(ind):
#         y[ind] = n[ind] * factorial(n[ind] - 1)

#     return y


if __name__ == "__main__":
    sources = [
        HermiteGaussian(m=3, n=2),
        TransverseElectric(lam=300 / 110, m=22, n=6, omf=1.06, rmin=8),
        TransverseElectric(lam=300 / 95, m=6, n=2, omf=1.06, rmin=2),
        GaussianBeam(lam=300 / 95, w0=10),
    ]

    for obj in sources:

        integrate_power(obj)
        process_fields(obj)
        obj.plot(scalars="||E||^2", cmap="jet")
