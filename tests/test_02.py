from glimmer.mom import MoM

from test_01 import src, m1

src.radiate(m1)


src.save('./temp/src.vtk')
m1.save('./temp/m1.vtk')

mom = MoM(pec=m1, k=src.k)

mom.plot_mesh()
mom.show_charts()
mom.save('./temp/pec.vtk')
