from glimmer import Problem
from glimmer.templates.gt import src, m1, m2, yz, vol

problem = Problem(source=src, optics=[m1, m2], probes=[vol, yz])
problem.solve()

plotter = problem.plot()
# plotter.camera_position = "xz"
plotter.show()
