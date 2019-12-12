import sys
from collections import defaultdict
import numpy as np


infile = sys.argv[1]
input = open(infile, "r")
data = input.read().split("\n")[1:-1]
input.close()


SYS_MOS = defaultdict(list)
SYS_SPK = defaultdict(list)
SPK_MOS = defaultdict(list)
SPK_SYS = defaultdict(list)

#spoofed,A07,LA_0028,LA_E_7151962.wav,5
for line in data:
	la_system = line.split(",")[1]
	speaker = line.split(",")[2]
	mos = int(line.split(",")[-1])
        SYS_MOS[la_system].append(mos)
        SYS_SPK[la_system].append(speaker)
        SPK_MOS[speaker].append(mos)
        SPK_SYS[speaker].append(la_system)

for k,v in SYS_MOS.items():
        la_system = k
        avg_mos = np.average(np.array(v))
        std_mos = np.std(np.array(v))
        print(la_system, avg_mos, std_mos, len(v))

for k,v in SYS_SPK.items():
        la_system = k
        unique_speakers = len(set(v))
        print(la_system, unique_speakers)

for k,v in SPK_MOS.items():
        speaker = k
        avg_mos = sum(v) / float(len(v))
        std_mos = np.std(np.array(v))
        print(speaker, avg_mos, std_mos, len(v))

for k,v in SPK_SYS.items():
        speaker = k
        systems = sorted(set(v))
        print(speaker, systems)
