import os
import shutil
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# ===== 配置 =====
SRC_BASE = "./Ldataset/RNAyuchuli"
DST_BASE = "./Ldataset/RNA200"
FASTA_SRC = os.path.join(SRC_BASE, "fasta")
NPY_SRC   = os.path.join(SRC_BASE, "npy")
FASTA_DST = os.path.join(DST_BASE, "fasta")
NPY_DST   = os.path.join(DST_BASE, "npy")
FASTA_EXTS = (".fasta", ".fa", ".fna", ".txt", ".seq")

os.makedirs(FASTA_DST, exist_ok=True)
os.makedirs(NPY_DST, exist_ok=True)

# ===== 准备任务列表 =====
tasks = []  # 每个元素为 (npy_src_path, fasta_src_path, npy_dst_path, fasta_dst_path)

# 先构建每个 batch 的 FASTA 文件集合（减少多次查找）
print("构建 FASTA 索引...")
fasta_map = {}  # batch_name -> set of filenames (不带扩展名)
for batch_name in os.listdir(FASTA_SRC):
    batch_path = os.path.join(FASTA_SRC, batch_name)
    if not os.path.isdir(batch_path):
        continue
    fnames = set()
    for f in os.listdir(batch_path):
        # 只记录主文件名（去掉扩展名）
        stem, ext = os.path.splitext(f)
        if ext in FASTA_EXTS:
            fnames.add(stem)
    fasta_map[batch_name] = fnames

print("生成复制任务列表...")
total = 0
for batch_name in os.listdir(NPY_SRC):
    npy_batch = os.path.join(NPY_SRC, batch_name)
    if not os.path.isdir(npy_batch):
        continue
    if batch_name not in fasta_map:
        print(f"警告：batch {batch_name} 无对应 FASTA，跳过")
        continue
    fasta_set = fasta_map[batch_name]
    for npy_file in os.listdir(npy_batch):
        if not npy_file.endswith(".npy"):
            continue
        stem = npy_file[:-4]
        if stem not in fasta_set:
            continue
        # 确定实际 FASTA 文件名（需要知道扩展名）
        # 这里需要在 FASTA batch 中查找实际存在的文件
        # 简便方法：尝试扩展名
        fasta_src = None
        for ext in FASTA_EXTS:
            candidate = os.path.join(FASTA_SRC, batch_name, stem + ext)
            if os.path.isfile(candidate):
                fasta_src = candidate
                break
        if fasta_src is None:
            continue
        npy_src = os.path.join(npy_batch, npy_file)
        npy_dst = os.path.join(NPY_DST, npy_file)
        fasta_dst = os.path.join(FASTA_DST, os.path.basename(fasta_src))
        tasks.append((npy_src, fasta_src, npy_dst, fasta_dst))
        total += 1

print(f"共 {total} 个任务待复制。")

# ===== 复制函数（供多进程调用） =====
def copy_pair(args):
    npy_src, fasta_src, npy_dst, fasta_dst = args
    shutil.copy2(npy_src, npy_dst)
    shutil.copy2(fasta_src, fasta_dst)
    return 1

if __name__ == "__main__":
    start = time.time()
    # 使用进程数，一般为 CPU 核心数，但 IO 密集型可以适当增加
    num_workers = min(cpu_count() * 2, 16)  # 根据磁盘性能调整
    print(f"使用 {num_workers} 个进程并行复制...")
    with Pool(processes=num_workers) as pool:
        # 分批提交，避免内存爆炸（tasks 可能很大）
        batch_size = 10000
        copied = 0
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            results = pool.map(copy_pair, batch_tasks)
            copied += sum(results)
            elapsed = time.time() - start
            speed = copied / elapsed if elapsed > 0 else 0
            print(f"已复制 {copied}/{total}，用时 {elapsed/60:.1f} 分钟，速度 {speed:.1f} 对/秒")
    elapsed = time.time() - start
    print(f"\n完成！总共复制 {copied} 对，总用时 {elapsed/60:.1f} 分钟。")