"""run gaussian telescope demo"""

from glimmer import Gaussian, Mirror, Volume, Problem
import numpy as np

src = Gaussian(w0=10, lam=3, num_lam=3, num_waist=2)
src.rotate_z(5)
src.rotate_y(3)

zR = np.pi * src.w0**2 / src.lam

s = zR / 2
c = np.cos(np.pi / 4)

L = src.num_waist * src.w0 * (1 + (s / zR) ** 2) ** 0.5

options = {"L": (L, L / c), "dL": src.lam / src.num_lam, "f": (s * c, s / c)}

m1 = Mirror(**options)
m1.rotate_x(45)
m1.translate([0, 0, s])

m2 = Mirror(**options)
m2.rotate_x(-45)
m2.translate([0, 2 * s, s])

yz = Volume(d=1, ylim=(-L / 2, 2 * s + L / 2), zlim=(0, s + L / 2))
vol = Volume(
    d=3, xlim=(-L / 2, L / 2), ylim=(-L / 2, 2 * s + L / 2), zlim=(0, s + L / 2)
)

problem = Problem(source=src, optics=[m1, m2], probes=[yz, vol])
problem.solve()
problem.plot().show()
