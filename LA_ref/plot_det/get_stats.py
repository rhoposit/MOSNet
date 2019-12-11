import sys
from collections import defaultdict


infile = sys.argv[1]
input = open(infile, "r")
data = input.read().split("\n")
input.close()


SYS_MOS = defaultdict(list)
SYS_SPK = defaultdict(list)

#spoofed,A07,LA_0028,LA_E_7151962.wav,5
for line in data:
	la_system = line.split(",")[1]
	speaker = line.split(",")[2]
	mos = line.split(",")[-1]
