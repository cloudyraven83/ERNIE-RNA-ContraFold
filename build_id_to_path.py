import os
import h5py
import pickle

h5_files = [
    os.path.expanduser("~/ERNIE-RNA/Ldataset/RNA200/contact_maps.h5"),
    os.path.expanduser("~/ERNIE-RNA/Ldataset/RNA200/contact_maps_part2.h5"),
    os.path.expanduser("~/ERNIE-RNA/Ldataset/RNA200/contact_maps_part3.h5"),
]

id_to_path = {}
for path in h5_files:
    print(f"正在扫描 {path} ...")
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            id_to_path[key] = path
    print(f"已扫描 {len(id_to_path)} 个ID")

# 保存为 pickle
out_path = os.path.expanduser("~/ERNIE-RNA/Ldataset/RNA200/id_to_path.pkl")
with open(out_path, 'wb') as f:
    pickle.dump(id_to_path, f)
print(f"映射已保存至 {out_path}")