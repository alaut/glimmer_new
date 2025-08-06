from glimmer.templates.gt import src, yz, vol, m1, m2
from glimmer.mom import MoM


mom = MoM(src + m1 + m2, k=src.k, probes=[src, m1, m2, vol, yz])

mom.solve()
mom.post()

plotter = mom.plot()
# plotter.camera_position = "xz"
plotter.show()
