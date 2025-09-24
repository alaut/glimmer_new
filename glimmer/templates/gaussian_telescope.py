from glimmer import Gaussian, Mirror, Grid, add_field
import numpy as np


from scipy.constants import milli

# milli = 1
lam = 3 * milli
w0 = 10 * milli

num_waist = 2
num_lam = 3

src = Gaussian(w0=w0, lam=lam, num_lam=num_lam, num_waist=num_waist)
# src.rotate_x(3, inplace=True, transform_all_input_vectors=True)

zR = np.pi * w0**2 / lam
s = zR / 2
c = np.cos(np.pi / 4)

L = num_waist * w0 * (1 + (s / zR) ** 2) ** 0.5

options = {"L": (L, L / c), "dL": lam / num_lam, "f": (s * c, s / c)}

m1 = Mirror(**options)
m1.rotate_x(45, inplace=True)
m1.translate([0, 0, s], inplace=True)

m2 = Mirror(**options)
m2.rotate_x(-45, inplace=True)
m2.translate([0, 2 * s, s], inplace=True)

add_field(m1, key="E")
add_field(m2, key="E")

optics = [
    m1,
    m2,
]

pec = src + m1 + m2

vol = Grid(ds=pec, d=1 * milli, scale=1.2)

yz = Grid(ds=pec, xlim=0, d=1 / 4 * milli, scale=1.1)
xy = Grid(xlim=(-L / 2, L / 2), ylim=(-L / 2, L / 2), d=1 / 4 * milli)

xz = xy.rotate_x(90).translate([0, s, s])
xy1 = xy.translate([0, 0, 5 * milli])
xy2 = xy.translate([0, 2 * s, 5 * milli])

probes = [
    # vol,
    yz,
    xy1,
    xz,
    xy2,
]
