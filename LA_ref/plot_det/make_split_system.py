import sys
from collections import defaultdict
import random

random.seed(1984)


infile = sys.argv[1] # LA_mos.csv
input = open(infile, "r")
data = input.read().split("\n")[1:-1]
input.close()

M = ['A08', 'A17', 'A09', 'A13', 'A18', 'A16', 'A11']
F = ['A19', 'A07', 'A14', 'A12', 'A15', 'A10']

total_M = len(M)
total_F = len(F)
print(total_M, total_F)

train_M_num = int(total_M * 0.65)+1
train_F_num = int(total_F * 0.65)+1

valid_M_num = int(total_M * 0.15)
valid_F_num = int(total_F * 0.15)+1

test_M_num = int(total_M * 0.2)
test_F_num = int(total_F * 0.2)


print("Poor:", train_M_num, valid_M_num, test_M_num, (train_M_num+valid_M_num+test_M_num))
print("Good:", train_F_num, valid_F_num, test_F_num, (train_F_num+valid_F_num+test_F_num))

train_M, valid_M, test_M = {}, {}, {}
train_F, valid_F, test_F = {}, {}, {}

F_count = 0
M_count = 0
for f in F:
    F_count += 1
    if F_count <= train_F_num:
        train_F[f] = []
    elif F_count <= train_F_num+valid_F_num:
        valid_F[f] = []
    elif F_count <= train_F_num+valid_F_num+test_F_num:
        test_F[f] = []
    
for m in M:
    M_count += 1
    if M_count <= train_M_num:
        train_M[m] = []
    elif M_count <= train_M_num+valid_M_num:
        valid_M[m] = []
    elif M_count <= train_M_num+valid_M_num+test_M_num:
        test_M[m] = []

print("Good disjoint systems")
print("Train", list(train_F.keys()))
print("Valid", list(valid_F.keys()))
print("Test", list(test_F.keys()))

print("Poor disjoint systems")
print("Train", list(train_M.keys()))
print("Valid", list(valid_M.keys()))
print("Test", list(test_M.keys()))


random.shuffle(data)
for line in data:
    items = line.split(",")
    user = items[0]
    audio = items[1]
    MOS = items[2]
    system = items[3]
    speakerID = items[4]
    outstring = audio+","+MOS
    
    #female speaker
    if system in F:
        if system in train_F:
            train_F[system].append(outstring)
        elif system in valid_F:
            valid_F[system].append(outstring)
        elif system in test_F:
            outstring = speakerID+","+system+","+outstring
            test_F[system].append(outstring) # keep system for eval

    #male speaker
    if system in M:
        if system in train_M:
            train_M[system].append(outstring)
        elif system in valid_M:
            valid_M[system].append(outstring)
        elif system in test_M:
            outstring = speakerID+","+system+","+outstring
            test_M[system].append(outstring) # keep system for eval

TRAIN,VALID,TEST = [],[],[]
c = 0
for k,v in train_F.items():
    TRAIN.extend(v)
    c += len(v)
print("F-train", c)
c = 0
for k,v in train_M.items():
    TRAIN.extend(v)
    c += len(v)
print("M-train", c)
c = 0

for k,v in valid_F.items():
    VALID.extend(v)
    c += len(v)
print("F-valid", c)
c = 0
for k,v in valid_M.items():
    VALID.extend(v)
    c += len(v)
print("M-valid", c)
c = 0

for k,v in test_F.items():
    TEST.extend(v)
    c += len(v)
print("F-test", c)
c = 0
for k,v in test_M.items():
    TEST.extend(v)
    c += len(v)
print("M-test",c)
c = 0

trainfile = "train_list.txt"
validfile = "valid_list.txt"
testfile = "test_list.txt"

output = open(trainfile, "w")
output.write("\n".join(TRAIN))
output.close()

output = open(validfile, "w")
output.write("\n".join(VALID))
output.close()

output = open(testfile, "w")
output.write("\n".join(TEST))
output.close()
