import os
from collections import Counter

real_list = open("20170817_LOBI_word_root_v3.txt").read().split("\n")
gen_list = os.listdir("gen_res")

real_list = [x.split("        ")[1].split("    ") for x in real_list if x != ""]

gen_list = [x.split(".")[0].split("-") for x in gen_list]

real_list = [Counter(x) for x in real_list]
gen_list = [Counter(x) for x in gen_list]

res = []

for i in real_list:
    for j in gen_list:
        if i == j:
            res.append("-".join(k for k in i))

print(sorted(list(set(res))))