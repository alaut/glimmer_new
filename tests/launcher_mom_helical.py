from scipy.constants import milli, degree
import numpy as np

from glimmer.sources import TransverseElectric

from glimmer.mom import Solver
from glimmer import add_field, process_fields, Grid

from glimmer.launcher import Launcher


milli = 1

lam = 300 / 95 * milli

num_lam = 3
launcher = Launcher(dL=lam / num_lam, prefix="./temp/launcher/mom")

add_field(launcher, key="E")
add_field(launcher, key="H")

t = np.atan2(launcher.points[..., 1], launcher.points[..., 0]) / degree
t = np.unique(np.round(t, 1))

source = TransverseElectric(
    lam=lam,
    m=6,
    n=2,
    a=6.5 * milli,
    rmin=2 * milli,
    num_lam=num_lam,
    nt=t.size + 1,
)

pec = launcher + source

vol = Grid(ds=pec, scale=1.25, d=1 / 2 * milli)

process_fields(pec)

problem = Solver(
    ds=pec,
    lam=lam,
    probes=[vol],
    tolerance=0.1 * milli,
    mode="cupy",
)

problem.tri.save(f"./temp/launcher.stl")
# problem.plot().show()

problem.solve()
problem.save(f"./temp/launcher/mom")

problem.plot().show()
