from glimmer.templates import gaussian_telescope as gt
from glimmer.mom import Solver
from glimmer.tools import integrate_power
from glimmer import Timer

prefix = "./temp/gt/mom"


def main(mode="cupy-sparse"):

    solver = Solver(
        ds=gt.pec,
        lam=gt.lam,
        probes=gt.probes,
        mode=mode,
    )

    # solver.plot().show()
    solver.solve()

    print(time.elapsed)

    # for obj in solver.probes:
    #     integrate_power(obj)

    # solver.save(prefix)
    # plotter = solver.plot()

    # plotter.show()


import pandas as pd

if __name__ == "__main__":

    modes = ["cupy-sparse", "cupy", "numpy-sparse", "tensorflow", "pytorch"]
    df = pd.DataFrame({"time": 0}, index=modes)

    for mode in modes:

        with Timer(mode) as time:
            main(mode)

        df["time"][mode] = time.elapsed.total_seconds()
    df = df.sort_values(by="time")
    df.to_excel(f"{prefix}runtime.xlsx")
    print(df)
