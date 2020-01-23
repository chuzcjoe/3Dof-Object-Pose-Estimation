import argparse
import math
import numpy as np
import re

# PART I: Information extraction
parser = argparse.ArgumentParser(description='Convert Yaw Pitch Yaw to frontal and right vectors')
parser.add_argument('filename', type=str, nargs=1, help='Specify your filename')
args = parser.parse_args()

filename = ''
if args.filename is not None:
    filename = args.filename[0]

dst = []
cnt = 0
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        if "LogBlueprintUserMessages: [Dewar_Blueprint_509]" in line:
            cnt += 1
            if cnt % 2:
                x_beg = line.find('X=')
                dst.append(line[x_beg: -1])
            elif not cnt % 2:
                p_beg = line.find('P=')
                dst.append(line[p_beg: -1])
        else:
            continue


dst_clean = [re.sub('[A-Z=]','',x.rstrip()) for x in dst]


i = 0
while i <= len(dst_clean)-1:
    save_path = './angles/img_{:06}.txt'.format(i//4)
    with open(save_path, 'w') as f:
        f.write(dst_clean[i] +'\n')
        f.write(dst_clean[i+1]+'\n')
        f.write(dst_clean[i+2]+'\n')
        f.write(dst_clean[i+3]+'\n')
    i += 4

print("done converting raw data to separate txt files.")
"""
dst_file = "raw_labels.txt"
with open(dst_file, 'a') as f:
    for i, line in enumerate(dst_clean):
        if i != len(dst_clean) - 1:
            f.write(line+'\n')
        else:
            f.write(line)
"""