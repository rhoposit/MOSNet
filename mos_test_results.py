import glob
import sys, os
from collections import defaultdict
import numpy as np

F = glob.glob("results/*.csv")
print(F)

SPK_MOS = defaultdict(list)
ANN_MOS = defaultdict(list)
agg_out_file = "test_file.csv"
AGG = []


# test1,5876-hvd_051.wav," 4 ",C1,23514
for f in F:
    ann = f.split(".")[0]
    print(f)
    input = open(f, "r")
    data = input.read().split("\n")[1:-1]
    input.close()
    print(len(data))
    for line in data:
        items = line.split(",")
        spk = items[1].split("-")[0]
        mos = int(items[2].split(" ")[1])
        fname = items[1]
        SPK_MOS[spk].append(mos)
        ANN_MOS[ann].append(mos)

        newline = spk+",ophelia,"+fname+","+str(mos)
        AGG.append(newline)

output = open(agg_out_file, "w")
output.write("\n".join(AGG))
output.close()


for ann,moslist in ANN_MOS.items():
    avg_mos = np.mean(np.array(moslist))
    min_mos = np.amin(np.array(moslist))
    max_mos = np.amax(np.array(moslist))
    std_mos = np.std(np.array(moslist))
    print(ann,avg_mos, std_mos, min_mos, max_mos)



for spk,moslist in SPK_MOS.items():
    avg_mos = np.mean(np.array(moslist))
    print(spk,avg_mos)
