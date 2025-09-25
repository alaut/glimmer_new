from glimmer.templates import gaussian_telescope as gt
from glimmer.mom import Solver
from glimmer.tools import integrate_power

solver = Solver(ds=gt.pec, lam=gt.lam, probes=gt.probes)

# solver.plot().show()
solver.solve()


for obj in solver.probes:
    integrate_power(obj)

solver.save("./temp/gt/mom")
plotter = solver.plot()


plotter.show()
