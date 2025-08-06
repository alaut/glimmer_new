from glimmer.mom import MoM
import pyvista as pv


from scipy.constants import milli
from glimmer.templates.gt import src, m1, m2, yz, vol

pv.global_theme.cmap = "jet"


def main(run=True, post=True, plot=True):
    """run EFIE MoM RWG solver demo problem"""

    if run:
        mom = MoM(src + m1 + m2, k=src.k, deg=2)
        mom.solve()
        mom.save("mom.vtk")
    else:
        vtk = pv.read("mom.vtk")
        mom = MoM(vtk, k=src.k)

    if post:
        mom.radiate(yz, chunks=16)
        yz.save("yz.vtk")
    else:
        yz = pv.read("yz.vtk")

    if post:
        mom.radiate(vol, chunks=16)
        vol.save("vol.vtk")
    else:
        vol = pv.read("vol.vtk")

    if plot:
        plotter = pv.Plotter()
        plotter.add_mesh(mom, scalars="|J|^2")
        plotter.add_volume(vol, scalars="|E|^2", opacity_unit_distance=milli)
        plotter.add_mesh(yz, scalars="|E|^2")
        plotter.show_grid()
        plotter.show()


if __name__ == "__main__":

    main(run=False, post=False, plot=True)
