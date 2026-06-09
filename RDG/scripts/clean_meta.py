import os
meta_file = "/home/adam/2tb-workspace/RDNU/RDG/basicsr/data/meta_info/m3vir_train_meta_info.txt"
with open(meta_file, 'r') as f:
    lines = f.readlines()

valid_lines = []
for line in lines:
    if "TRACK" in line: continue
    parts = line.strip().split()
    if len(parts) >= 5:
        frames = int(parts[2])
        if frames >= 6:
            valid_lines.append(line)

with open(meta_file, 'w') as f:
    f.writelines(valid_lines)
print(f"Cleaned {len(lines)} to {len(valid_lines)} lines")
