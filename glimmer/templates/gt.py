import pyvista as pv
from glimmer import Gaussian, Mirror
import numpy as np


from glimmer.tools import poly2grid


src = Gaussian(w0=10, lam=3, num_lam=3, num_waist=2)
print(src)
# src.plot()
src.save("./temp/source.vtk")

src = src.rotate_z(5)
src.rotate_y(3)

zR = np.pi * src.w0**2 / src.lam

s = zR / 2
c = np.cos(np.pi / 4)

L = src.num_waist * src.w0 * (1 + (s / zR) ** 2) ** 0.5

options = {"L": (L, L / c), "dL": src.lam / src.num_lam, "f": (s * c, s / c)}

m1 = Mirror(**options)
m1.rotate_x(45)
m1.translate([0, 0, s])

m2 = Mirror(**options)
m2.rotate_x(-45)
# m2.rotate_y(-5)
m2.translate([0, 2 * s, s])

pec = src + m1 + m2

vol = poly2grid(pec, d=2, l=10)
vol.chunks = 32
yz = poly2grid(pec, d=1 / 2, x=0, l=10)
yz.chunks = 32


def preview():
    """Preview the Gaussian telescope setup."""
    plotter = pv.Plotter()
    for obj in [src, m1, m2, vol, yz]:
        plotter.add_mesh(obj, style="wireframe", color="black", line_width=1)
    plotter.show()


if __name__ == "__main__":

    preview()
