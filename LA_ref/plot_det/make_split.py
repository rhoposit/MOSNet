import sys
from collections import defaultdict
import random

random.seed(1984)


infile = sys.argv[1] # LA_mos.csv
input = open(infile, "r")
data = input.read().split("\n")[1:-1]
input.close()

M = ['LA_0001','LA_0002','LA_0003','LA_0005','LA_0007','LA_0011','LA_0013','LA_0015','LA_0018','LA_0021','LA_0023','LA_0025','LA_0028','LA_0030','LA_0032','LA_0036','LA_0038','LA_0040','LA_0044','LA_0046','LA_0048']
F = ['LA_0004','LA_0006','LA_0008','LA_0009','LA_0010','LA_0012','LA_0014','LA_0016','LA_0017','LA_0019','LA_0020','LA_0022','LA_0024','LA_0026','LA_0027','LA_0029','LA_0031','LA_0033','LA_0034','LA_0035','LA_0037','LA_0039','LA_0041','LA_0042','LA_0043','LA_0045','LA_0047']

total_M = len(M)
total_F = len(F)
print(total_M, total_F)

train_M_num = int(total_M * 0.65)+1
train_F_num = int(total_F * 0.65)+1

valid_M_num = int(total_M * 0.15)
valid_F_num = int(total_F * 0.15)

test_M_num = int(total_M * 0.2)
test_F_num = int(total_F * 0.2)


print("Male:", train_M_num, valid_M_num, test_M_num, (train_M_num+valid_M_num+test_M_num))
print("Female:", train_F_num, valid_F_num, test_F_num, (train_F_num+valid_F_num+test_F_num))

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

print("Female disjoint speakers")
print("Train", list(train_F.keys()))
print("Valid", list(valid_F.keys()))
print("Test", list(test_F.keys()))

print("Male disjoint speakers")
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
    if speakerID in F:
        if speakerID in train_F:
            train_F[speakerID].append(outstring)
        elif speakerID in valid_F:
            valid_F[speakerID].append(outstring)
        elif speakerID in test_F:
            outstring = speakerID+","+system+","+outstring
            test_F[speakerID].append(outstring) # keep system for eval

    #male speaker
    if speakerID in M:
        if speakerID in train_M:
            train_M[speakerID].append(outstring)
        elif speakerID in valid_M:
            valid_M[speakerID].append(outstring)
        elif speakerID in test_M:
            outstring = speakerID+","+system+","+outstring
            test_M[speakerID].append(outstring) # keep system for eval

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
