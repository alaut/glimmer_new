from glimmer.sources import *

sources = [
    HermiteGaussian(m=3, n=2),
    TransverseElectric(lam=300 / 110, m=22, n=6, omf=1.06, rmin=8),
    TransverseElectric(lam=300 / 95, m=6, n=2, omf=1.06, rmin=2),
    GaussianBeam(lam=300 / 95, w0=10),
]

for obj in sources:

    integrate_power(obj)
    process_fields(obj)
    obj.plot(scalars="||E||^2", cmap="jet")
