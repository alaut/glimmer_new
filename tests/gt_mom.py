from glimmer import Gaussian, Mirror
import numpy as np

from scipy.constants import milli, c, giga

from glimmer.mom import MoM, poly2grid
import pyvista as pv


def main(run=True, post=True):
    """run EFIE MoM RWG solver demo problem"""

    src = Gaussian(w0=10 * milli, lam=c / (95 * giga), num_lam=3, num_waist=2)
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
    m2.rotate_z(15)
    m2.translate([0, 2 * s, s])

    if run:
        grid = src + m1 + m2

        mom = MoM(grid, k=src.k, deg=2)

        mom.solve()

        print("saving vtk ...")
        mom.save("mom.vtk")
    else:
        vtk = pv.read("mom.vtk")
        mom = MoM(vtk, k=src.k)

    if post:
        yz = poly2grid(mom, d=milli / 2, x=0, l=10 * milli)
        mom.radiate(yz, chunks=16)
        yz.save("yz.vtk")
    else:
        yz = pv.read("yz.vtk")

    if post:
        vol = poly2grid(mom, d=milli, l=10 * milli)
        mom.radiate(vol, chunks=16)
        vol.save("vol.vtk")
    else:
        vol = pv.read("vol.vtk")

    plotter = pv.Plotter()
    plotter.add_mesh(mom, scalars="|J|^2", cmap="jet")
    plotter.add_volume(vol, scalars="|E|^2", opacity_unit_distance=milli, cmap="jet")
    plotter.add_mesh(yz, scalars="|E|^2", cmap="jet")
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":

    main(run=True, post=True)
