import os
from pyvista import StructuredGrid, Plotter
from dataclasses import dataclass
import numpy as np
import cupy as cp


def unpack(f):

    try:
        fx, fy = f
    except:
        fx = fy = f
    return fx, fy


def span(L, dL):

    Lx, Ly = unpack(L)

    return np.meshgrid(
        np.linspace(-Lx / 2, Lx / 2, int(Lx / dL) + 1),
        np.linspace(-Ly / 2, Ly / 2, int(Ly / dL) - 1),
    )


def logclip(A0, clip=99.5, dBmin=-30):

    Amax = np.nanpercentile(A0, clip)
    A0 = np.clip(A0, 0, Amax)

    with np.errstate(divide="ignore", invalid="ignore"):
        A0dB = np.clip(10 * np.log10(A0 / Amax), dBmin, 0)

    return A0, A0dB


def StrattonChu(r1, E1, H1, k1, r2, mode="radiate", eta=377, num=5):

    E1 = cp.asarray(E1)
    H1 = cp.asarray(H1)

    drdu, drdv = cp.gradient(r1, axis=(0, 1))
    dSvec = cp.cross(drdv, drdu, axis=-1)

    dS1 = cp.linalg.norm(dSvec, axis=-1, keepdims=True)
    n1 = dSvec / dS1

    match mode:
        case "reflect":
            E1 = 2 * cp.sum(E1 * n1, axis=-1, keepdims=True) * n1 - E1
            H1 = H1 - 2 * cp.sum(H1 * n1, axis=-1, keepdims=True) * n1
        case "negate":
            E1 = -E1
            H1 = -H1

    r = cp.asarray(r2[..., None, None, None, :] - r1)

    R = cp.linalg.norm(r, axis=-1, keepdims=True)

    G = cp.exp(1j * k1 * R) / (4 * cp.pi * R)
    dG = r / R * G * (1 / R - 1j * k1)

    nxH = cp.cross(n1, H1, axis=-1)
    nxE = cp.cross(n1, E1, axis=-1)

    n_dot_E = cp.sum(n1 * E1, axis=-1, keepdims=True)
    n_dot_H = cp.sum(n1 * H1, axis=-1, keepdims=True)

    dE = cp.cross(nxE, dG, axis=-1) + 1j * k1 * nxH * G * eta + n_dot_E * dG
    dH = cp.cross(nxH, dG, axis=-1) - 1j * k1 * nxE * G / eta + n_dot_H * dG

    dE[R[..., 0] < 1 / k1 / num] = cp.nan
    dH[R[..., 0] < 1 / k1 / num] = cp.nan

    E2 = cp.nansum(dS1 * dE, axis=(-4, -3, -2)).get()
    H2 = cp.nansum(dS1 * dH, axis=(-4, -3, -2)).get()

    return E2, H2


class Grid(StructuredGrid):

    chunks = 1

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.set_fields()

    def round(self, decimals=3):

        for k, v in self.__dict__.items():
            if "_" not in k and v is not None:
                try:
                    self.__dict__[k] = tuple(float(round(x, decimals)) for x in v)
                except:
                    self.__dict__[k] = float(round(v, decimals))

    def translate(
        self, translation, inplace=True, transform_all_input_vectors=True, **kwargs
    ):
        return super().translate(
            translation,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def rotate_x(self, angle, inplace=True, transform_all_input_vectors=True, **kwargs):
        return super().rotate_x(
            angle,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def rotate_y(self, angle, inplace=True, transform_all_input_vectors=True, **kwargs):
        return super().rotate_y(
            angle,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def rotate_z(self, angle, inplace=True, transform_all_input_vectors=True, **kwargs):
        return super().rotate_z(
            angle,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def transform(
        self, transformation, inplace=True, transform_all_input_vectors=True, **kwargs
    ):
        return super().transform(
            transformation,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def rotation(
        self, rotation, inplace=True, transform_all_input_vectors=True, **kwargs
    ):
        return super().rotation(
            rotation,
            inplace=inplace,
            transform_all_input_vectors=transform_all_input_vectors,
            **kwargs,
        )

    def radiate(self, other, mode="radiate"):

        print(
            f"{mode} {self.__repr__()} onto {other.chunks} chunks of {other.__repr__()}"
        )

        E1, H1 = self.get_fields()

        chunks = np.array_split(
            np.reshape(other.points_matrix, (-1, 3)), other.chunks, axis=0
        )

        inputs = {
            "r1": self.points_matrix,
            "E1": E1,
            "H1": H1,
            "k1": self.k,
            "mode": mode,
        }

        F = [StrattonChu(**inputs, r2=chunk) for chunk in chunks]

        E2, H2 = zip(*F)

        E2 = np.reshape(np.concatenate(E2), other.points_matrix.shape)
        H2 = np.reshape(np.concatenate(H2), other.points_matrix.shape)

        E, H = other.get_fields()

        other.k = self.k
        other.set_fields(E + E2, H + H2)

    def negate(self, probe, **kwargs):
        self.radiate(probe, mode="negate", **kwargs)

    def reflect(self, probe, **kwargs):
        self.radiate(probe, mode="reflect", **kwargs)

    def set_fields(self, E=None, H=None):

        E = np.zeros_like(self.points_matrix) if E is None else E
        H = np.zeros_like(self.points_matrix) if H is None else H

        E = np.reshape(E, shape=(-1, 3), order="F")
        H = np.reshape(H, shape=(-1, 3), order="F")

        S = 0.5 * np.real(np.cross(E, np.conj(H), axis=-1))

        self["Er"] = E.real
        self["Ei"] = E.imag
        self["Hr"] = H.real
        self["Hi"] = H.imag
        self["Sr"] = S.real
        self["Si"] = S.imag

        E2 = np.linalg.norm(E, axis=-1) ** 2
        H2 = np.linalg.norm(H, axis=-1) ** 2
        S0 = np.linalg.norm(S, axis=-1)

        self["|E|^2"], self["|E|^2 (dB)"] = logclip(E2)
        self["|H|^2"], self["|E|^2 (dB)"] = logclip(H2)
        self["|<S>|"], self["|<S>| (dB)"] = logclip(S0)

        self.set_active_vectors("Sr")
        self.set_active_scalars("|<S>|")

    def get_fields(self):

        E = self["Er"] + 1j * self["Ei"]
        H = self["Hr"] + 1j * self["Hi"]

        E = np.reshape(E, self.points_matrix.shape, order="F")
        H = np.reshape(H, self.points_matrix.shape, order="F")

        return E, H


@dataclass
class Volume(Grid):

    xlim: tuple = 0
    ylim: tuple = 0
    zlim: tuple = 0

    d: float = 1

    chunks: int = 1

    def __post_init__(self, **kwargs):

        x, y, z = np.meshgrid(
            self.limits(self.xlim),
            self.limits(self.ylim),
            self.limits(self.zlim),
        )

        super().__init__(x, y, z, **kwargs)

    def limits(self, lim):
        try:
            return np.linspace(*lim, int((lim[1] - lim[0]) / self.d))
        except:
            return float(lim)


@dataclass
class Gaussian(Grid):

    w0: float
    lam: float

    num_lam: float = 3
    num_waist: float = 3

    P0: float = 1
    Z0: float = 377

    def __post_init__(self):

        self.k = 2 * np.pi / self.lam

        wx, wy = unpack(self.w0)

        x, y = span((wx * self.num_waist, wy * self.num_waist), self.lam / self.num_lam)

        super().__init__(*np.broadcast_arrays(x, y, 0))

        x = self.points_matrix[..., 0]
        y = self.points_matrix[..., 1]

        I0 = 2 * self.P0 / (np.pi * wx * wy)
        A = (I0 * self.Z0) ** 0.5 * np.exp(-(x**2) / wx**2 - y**2 / wy**2)

        E = A[..., None] * np.array([1, 0, 0])
        H = A[..., None] * np.array([0, 1, 0]) / self.Z0

        self.set_fields(E, H)


@dataclass
class Mirror(Grid):

    L: tuple
    dL: float

    f: float = None

    def __post_init__(self, **kwargs):

        x, y = span(self.L, self.dL)

        if self.f is not None:
            fx, fy = unpack(self.f)

            z = -(x**2) / (4 * fx) - y**2 / (4 * fy)
        else:
            z = np.zeros_like(x)

        super().__init__(x, y, z, **kwargs)

        self.round()


@dataclass
class Problem:

    source: object
    optics: list = ()
    probes: list = ()

    cmap: str = "jet"

    interactive: bool = False

    def solve(self):

        for obj in [self.source, *self.optics, *self.probes]:
            obj.round()

        sources = [self.source.radiate]

        for optic in self.optics:
            sources[-1](optic)
            sources.extend([optic.negate, optic.reflect])

        for probe in self.probes:
            for source in sources:
                source(probe)

    def update_scene(self):

        for obj, actor in self.actors:
            obj.transform(actor.GetMatrix())

            actor.SetUserTransform(None)
            actor.SetPosition(0, 0, 0)
            actor.SetOrientation(0, 0, 0)
            actor.SetScale(1, 1, 1)

        for obj in [*self.probes, *self.optics]:
            obj.set_fields()

        self.solve()

        for i, (obj, actor) in enumerate(self.actors):
            if obj.dimensionality == 3:
                self.plotter.remove_actor(actor)
                new_actor = self.plotter.add_volume(obj, clim=self.clim, cmap=self.cmap)
                self.actors[i] = (obj, new_actor)

        self.plotter.render()

    def plot(self):

        self.plotter = Plotter()

        objects = [self.source, *self.optics, *self.probes]

        scalars = np.concat([np.ravel(obj.active_scalars) for obj in objects])

        self.clim = (np.nanmin(scalars), np.nanmax(scalars))

        def add_object(obj):
            try:
                return self.plotter.add_volume(obj, clim=self.clim, cmap=self.cmap)
            except:
                return self.plotter.add_mesh(obj, clim=self.clim, cmap=self.cmap)

        self.actors = [(obj, add_object(obj)) for obj in objects]

        self.plotter.enable_parallel_projection()
        self.plotter.add_axes_at_origin()
        self.plotter.show_grid()

        self.plotter.add_key_event("u", self.update_scene)
        self.plotter.add_key_event("i", self.toggle_interactive)

        return self.plotter

    def toggle_interactive(self):

        if self.interactive:
            self.plotter.enable_trackball_style()
            self.interactive = False
        else:
            self.plotter.enable_trackball_actor_style()
            self.interactive = True

    def save(self, name):

        os.makedirs(os.path.dirname(name), exist_ok=True)

        self.source.save(f"{name}.source.vtk")

        for i, optic in enumerate(self.optics):
            optic.save(f"{name}.optic.{i}.vtk")

        for i, probe in enumerate(self.probes):
            probe.save(f"{name}.probe.{i}.vtk")


if __name__ == "__main__":

    src = Gaussian(w0=10, lam=3, num_lam=3, num_waist=2)
    src.rotate_z(5)
    src.rotate_y(3)

    zR = np.pi * src.w0**2 / src.lam

    s = zR / 2
    c = np.cos(np.pi / 4)

    L = src.num_waist * src.w0 * (1 + (s / zR) ** 2) ** 0.5

    options = {"L": (L, L / c), "dL": src.lam / src.num_lam, "f": (s * c, s / c)}

    m1 = Mirror(**options)
    m1.rotate_x(45)
    m1.translate([0, 0, s])
    m1.round()

    m2 = Mirror(**options)
    m2.rotate_x(-45)
    m2.translate([0, 2 * s, s])
    m2.round()

    yz = Volume(d=1, ylim=(-L / 2, 2 * s + L / 2), zlim=(0, s + L / 2))
    vol = Volume(
        d=3, xlim=(-L / 2, L / 2), ylim=(-L / 2, 2 * s + L / 2), zlim=(0, s + L / 2)
    )

    problem = Problem(source=src, optics=[m1, m2], probes=[yz, vol])
    problem.solve()
    problem.plot().show()
