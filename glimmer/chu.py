import cupy as cp

from scipy.constants import mu_0, epsilon_0
import numpy as np

eta = np.sqrt(mu_0 / epsilon_0)


def dot(A, B):
    return cp.sum(A * B, axis=-1, keepdims=True)


def cross(A, B):
    return cp.cross(A, B, axis=-1)


def norm(A):
    return cp.linalg.norm(A, axis=-1, keepdims=True)


def StrattonChu(r1, E1, H1, k1, r2, mode="radiate"):

    r1 = cp.asarray(r1)
    E1 = cp.asarray(E1)
    H1 = cp.asarray(H1)
    r2 = cp.asarray(r2)

    drdu, drdv = cp.gradient(r1, axis=(0, 1))
    dSvec = cp.cross(drdv, drdu, axis=-1)

    dS1 = norm(dSvec)
    n1 = dSvec / dS1

    match mode:
        case "reflect":
            E1 = 2 * dot(E1, n1) * n1 - E1
            H1 = H1 - 2 * dot(H1, n1) * n1
        case "negate":
            E1 = -E1
            H1 = -H1

    r = r2[..., None, None, :] - r1

    R = norm(r)

    G = cp.exp(1j * k1 * R) / (4 * cp.pi * R)
    dG = r * (G / R**2 - 1j * k1 * G / R)

    Js = cross(n1, H1)
    Ms = cross(E1, n1)

    dE = cross(dG, Ms) + 1j * k1 * Js * G * eta + dot(n1, E1) * dG
    dH = cross(Js, dG) + 1j * k1 * Ms * G / eta + dot(n1, H1) * dG

    E2 = cp.nansum(dS1 * dE, axis=(-3, -2))
    H2 = cp.nansum(dS1 * dH, axis=(-3, -2))

    return E2.get(), H2.get()


def EH_Gaussian(r1, wx, wy, P0=1):

    x, y, z = cp.moveaxis(cp.asarray(r1), -1, 0)

    I0 = 2 * P0 / (cp.pi * wx * wy)
    A = (I0 * eta) ** 0.5 * cp.exp(-(x**2) / wx**2 - y**2 / wy**2)

    E = A * cp.array([1, 0, 0])
    H = A * cp.array([0, 1, 0]) / eta

    return E.get(), H.get()


def EH_Hermite(r, l, m, c, wx, wy):
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    c = np.atleast_1d(c)

    mf = factorial(l)
    nf = factorial(m)

    x = r[..., 0]
    y = r[..., 1]

    # wx, wy = self.get_waists()

    Hm = hermite(l, 2**0.5 * x / wx)
    Hn = hermite(m, 2**0.5 * y / wy)

    A0 = Hm * Hn / np.sqrt(np.pi * wx * wy * 2.0 ** (l + m - 1) * mf * nf)

    X = np.exp(-(x**2) / wx**2)[..., None]
    Y = np.exp(-(y**2) / wy**2)[..., None]
    A = np.sum(c * A0 * X * Y, axis=-1)

    E = A[..., None] * np.array([1, 0, 0])
    H = A[..., None] * np.array([0, 1, 0]) / eta
    return E, H


def hermite(m, x):
    """evaluates hermite polynomial H_m(x)"""

    y = (m == 0) + 2 * (m == 1) * x[..., None]

    ind = m > 1

    if any(ind):

        h1 = 2 * x[..., None] * hermite(m[ind] - 1, x)
        h2 = 2 * (m[ind] - 1) * hermite(m[ind] - 2, x)

        y[..., ind] = h1 - h2

    return y


def factorial(n):
    """returns factorial"""

    y = (n == 0).astype(float) + (n == 1).astype(float)

    ind = n > 1

    if any(ind):
        y[ind] = n[ind] * factorial(n[ind] - 1)

    return y
