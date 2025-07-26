"""run EFIE MoM RWG solver demo problem"""

from glimmer import Gaussian, Mirror
import numpy as np

from scipy.constants import milli, c, giga

from glimmer.mom import MoM
import pyvista as pv


def poly2grid(poly, d, x=None, y=None, z=None):

    xmin, xmax, ymin, ymax, zmin, zmax = poly.bounds

    nx = int((xmax - xmin) / d) + 1
    ny = int((ymax - ymin) / d) + 1
    nz = int((zmax - zmin) / d) + 1

    x = np.linspace(xmin, xmax, nx) if x is None else float(x)
    y = np.linspace(ymin, ymax, ny) if y is None else float(y)
    z = np.linspace(zmin, zmax, nz) if z is None else float(z)

    x, y, z = np.meshgrid(x, y, z)

    return pv.StructuredGrid(np.squeeze(x), np.squeeze(y), np.squeeze(z))


src = Gaussian(w0=10 * milli, lam=c / (95 * giga), num_lam=4, num_waist=2)
src.rotate_z(5)
src.rotate_y(3)

zR = np.pi * np.array(src.w0) ** 2 / src.lam

s = zR / 2
cs = np.cos(np.pi / 4)
w = src.w0 * (1 + (s / zR) ** 2) ** 0.5
L = src.num_waist * w

options = {
    "L": (L, L / cs),
    "dL": src.lam / src.num_lam,
    "f": (s * cs, s / cs),
}

m1 = Mirror(**options)
m1.rotate_x(45)
m1.rotate_z(3)
m1.translate([0, 0, s])

m2 = Mirror(**options)
# m2.rotate_x(-135)
m2.rotate_x(-70)
# m2.rotate_z(3)
m2.rotate_z(5)
m2.translate([0, 2 * s, s])

# src.radiate(m1)

# mom.plot_rwg_basis()
if False:
    grid = src + m1 + m2
    mom = MoM(grid, k=src.k)
    mom.plot_excitation()
    mom.solve()
    mom.save("mom.vtk")
else:
    vtk = pv.read("mom.vtk")
    mom = MoM(vtk, k=src.k)

# mom.plot_currents()
# mom.show_charts()

vol = poly2grid(mom, d=milli * 2)
yz = poly2grid(mom, d=milli / 2, x=0)

mom.radiate(vol)
mom.radiate(yz)

plotter = pv.Plotter()
plotter.add_mesh(mom, scalars="|J|^2", cmap="jet")
plotter.add_volume(vol, scalars="|E|^2", opacity_unit_distance=milli, cmap="jet")
plotter.add_mesh(yz, scalars="|E|^2", cmap="jet")
plotter.show_grid()
plotter.show()
