import pandas as pd
import numpy as np

#pd.formats.format.header_style = None

seeds =  [39, 42890, 3424, 23232]

"""
23232_intrinsic_mean.npy  3424_intrinsic_mean.npy  39_intrinsic_mean.npy  42890_intrinsic_mean.npy
23232_intrinsic_std.npy   3424_intrinsic_std.npy   39_intrinsic_std.npy   42890_intrinsic_std.npy
""" # change

index = []
vals = []
for seed in seeds:
    seed = str(seed)
    for t in ['intrinsic', 'extrinsic']:
        for what in ['mean', 'std']:
            val = (np.load("data/" + seed + "_" + t + "_" + what + ".npy"))
            index.append(("Rewards in Pendulum Dynamics", seed, t, what)) #, col))#, val))
            vals.append(val)
                #vals.append(val[col-1])


path = "data/ScienceFairDynamicsDataRev.xlsx"
index = pd.MultiIndex.from_tuples(index)
data = np.array(vals)
pop = pd.DataFrame(data, index=index, columns=[list(range(20))])
print(pop)
writer = pd.ExcelWriter(path, engine="xlsxwriter")
pop.to_excel(writer, 'Sheet1')
writer.save()

"""

writer = pd.ExcelWriter(path, engine="xlsxwriter")
pop = pd.Series(vals, index=index)
print(pop, "before")
index = pd.MultiIndex.from_tuples(index)
#print(index.levels)
pop = pop.reindex(index)
pop = pd.DataFrame(pop, index=index, columns=list(range(0, 20)))
print(pop, "after")
"""
