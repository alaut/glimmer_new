from glimmer.chu import Solver
from glimmer.templates import gaussian_telescope as gt
from glimmer.tools import integrate_power


solver = Solver(
    lam=gt.lam,
    source=gt.src,
    optics=[gt.m1, gt.m2],
    probes=[
        gt.vol,
        gt.yz,
        gt.xy1,
        gt.xz,
        gt.xy2,
    ],
)

# solver.plot().show()
solver.solve()

for obj in [solver.source, *solver.optics, *solver.probes]:
    integrate_power(obj)

solver.save("./temp/gt/chu")
solver.plot().show()
