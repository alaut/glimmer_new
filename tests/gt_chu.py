from glimmer import Problem
from glimmer.templates.gt import src, m1, m2, yz, vol

problem = Problem(source=src, optics=[m1, m2], probes=[yz, vol])
problem.solve()
problem.plot().show()
