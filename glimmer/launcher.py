import numpy as np
import pyvista as pv

import os
import shutil

from dataclasses import dataclass

from scipy.constants import milli

milli = 1


@dataclass
class Launcher(pv.PolyData):

    prefix: str

    m: int = 6
    n: int = 2

    a: float = 6.5 * milli
    L: float = 35 * milli
    Lcut: float = 15.75 * milli

    ccw: int = 0

    dL: float = 0.25 * milli

    zfg: str = "./data/FXF-KERI-0041_v03.opt.00"

    def __post_init__(self):

        # generate launcher mesh
        os.system(
            f'genmesh {self.a} {self.L} {self.Lcut} {self.dL} {self.ccw} "{self.prefix}"'
        )

        # apply perturbation to mesh points
        if self.zfg is not None:
            os.system(f'modgeom {self.ccw} "{self.prefix}" "{self.zfg}"')

        # load pts/con files
        pts = np.loadtxt(f"{self.prefix}.pts")
        con = np.loadtxt(f"{self.prefix}.con", dtype=int)

        # triangulate connectivity
        faces = np.hstack([np.full((con.shape[0], 1), 3), con]).ravel()

        super().__init__(pts, faces=faces)

        self.save(f"{self.prefix}.stl")


if __name__ == "__main__":
    plotter = pv.Plotter()
    dL = np.array([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5])
    for i, dl in enumerate(dL):
        tri = Launcher(prefix=f"./temp/launchers/{dl:0.3f} mm", dL=dl)
        tri.save
        plotter.add_mesh(tri)
    plotter.show()
    # tri.plot()
