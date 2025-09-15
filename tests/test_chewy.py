from glimmer.chewy import Gaussian, Mirror, Volume, solve
import numpy as np

w0 = 10
lam = 3
num_lam = 4

num_waist = 3

zR = np.pi * w0**2 / lam
s = zR / 2

L = num_waist * w0 * np.sqrt(1 + (s / zR) ** 2) * np.array([1, np.sqrt(2)])

src = Gaussian(w0, lam, num_waist=num_waist)
src.rotate_x(3, inplace=True, transform_all_input_vectors=True)

m1 = Mirror(L=L, dL=lam / num_lam, f=(s / 2**0.5, s * 2**0.5))
m1.rotate_x(135 + 90, inplace=True)
m1.translate([0, 0, s], inplace=True)

m2 = Mirror(L=L, dL=lam / num_lam, f=(s / 2**0.5, s * 2**0.5))
m2.rotate_x(135, inplace=True)
m2.translate([0, 2 * s, s], inplace=True)

vol = Volume(
    d=1.5,
    xlim=(-L[0] / 2, L[0] / 2),
    ylim=(-L[1] / 2, 2 * s + L[1] / 2),
    zlim=(-L[1] / 2, s + L[1] / 2),
)


plotter = solve(src, [m1, m2], probes=[vol], prefix="./temp/chewy/mb")
plotter.show_grid()
plotter.show()
