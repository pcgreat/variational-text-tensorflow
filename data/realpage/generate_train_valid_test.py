import random

with open("chattrans_skipthought.txt") as f:
    lines = f.readlines()

trains = []
valids = []
tests = []
for line in lines:
    p = random.random()
    if p < 0.8:
        trains.append(line)
    elif p < 0.9:
        valids.append(line)
    else:
        tests.append(line)
        
with open("train.txt", "w") as f:
    f.write("".join(trains))
with open("valid.txt", "w") as f:
    f.write("".join(valids))
with open("test.txt", "w") as f:
    f.write("".join(tests))
