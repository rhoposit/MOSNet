import sys
from collections import defaultdict
import numpy as np
import random
SEED = 1984
random.seed(SEED)
ITER = 1
from sklearn.metrics import mean_squared_error as MSE

infile = sys.argv[1] # TABLE
input = open(infile, "r")
data = input.read().split("\n")[1:-1]
input.close()

OUT = []
outstring = "user_ID,audio_sample,MOS,system_ID,speaker_ID"
OUT.append(outstring)
U = []
S = defaultdict(list)
SP = defaultdict(list)


for line in data:
        user = line.split(",")[1]
        mos = line.split(",")[-4]
        fname = line.split(",")[-5]
        system = line.split(",")[5].split("-")[-1]
        speaker = line.split(",")[5].split("-")[0]
        if system[0] == "A":
                outstring = user+","+fname+","+mos+","+system+","+speaker
                OUT.append(outstring)
                U.append(user)
                S[system].append(int(mos))
                SP[speaker].append(int(mos))
outfile = "LA_mos.csv"
output = open(outfile, "w")
out = "\n".join(OUT)
output.write(out)
output.close()
         

print(len(set(U)), len(set(U))/2)


OUT = []
outstring = "system_ID,mean"
OUT.append(outstring)
for k,v in S.items():
        system = k
        mean = np.mean(np.array(v))
        outstring = system+","+str(mean)
        OUT.append(outstring)
outfile = "LA_mos_system.csv"
output = open(outfile, "w")
out = "\n".join(OUT)
output.write(out)
output.close()


OUT = []
outstring = "speaker_ID,mean"
OUT.append(outstring)
for k,v in SP.items():
        speaker = k
        mean = np.mean(np.array(v))
        outstring = speaker+","+str(mean)
        OUT.append(outstring)
outfile = "LA_mos_speaker.csv"
output = open(outfile, "w")
out = "\n".join(OUT)
output.write(out)
output.close()
