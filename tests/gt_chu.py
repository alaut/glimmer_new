from glimmer.chu import Solver
from glimmer.templates.gt import src, m1, m2, yz, vol, lam, xy2, xy1, xz

solver = Solver(
    lam=lam,
    source=src,
    optics=[m1, m2],
    probes=[
        # vol,
        yz,
        xz,
        xy1,
        xy2,
    ],
)

# solver.plot().show()
solver.solve()
solver.save("./temp/chu-mb/gt")
solver.plot().show()
