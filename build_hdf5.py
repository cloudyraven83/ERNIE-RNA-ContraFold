import os
import re
import numpy as np
import h5py
from tqdm import tqdm
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# ===== 路径配置 =====
npy_dir = "/local3/home/lizhizhuo/ERNIE-RNA/Ldataset/RNA200/npy/"
fasta_path = "/local3/home/lizhizhuo/ERNIE-RNA/Ldataset/RNA200/RNA2M.fasta"

# 基础文件名（不含编号）
base_h5_name = "contact_maps"
h5_dir = "/local3/home/lizhizhuo/ERNIE-RNA/Ldataset/RNA200/"

# 每个分片文件最大数据集数量（达到此值后自动新建下一个分片）
MAX_DATASETS_PER_FILE = 800000 

NUM_THREADS = 8

# ===== 辅助函数：获取已有分片文件列表 =====
def get_existing_part_files():
    """返回已存在的分片文件路径列表，按编号排序（主文件视为 part0）"""
    pattern = re.compile(rf"^{re.escape(base_h5_name)}(?:\.h5|_part(\d+)\.h5)$")
    files = []
    for f in os.listdir(h5_dir):
        if not f.endswith('.h5'):
            continue
        match = pattern.match(f)
        if match:
            if match.group(1) is None:
                # 主文件，编号为0
                part_num = 0
            else:
                part_num = int(match.group(1))
            files.append((part_num, os.path.join(h5_dir, f)))
    files.sort(key=lambda x: x[0])
    return files

# ===== 获取所有已有 keys（从所有分片中） =====
def get_all_existing_keys(part_files):
    keys = set()
    for _, path in part_files:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                keys.update(f.keys())
    return keys

# ===== 确定当前要写入的文件路径 =====
def get_output_file(part_files):
    if not part_files:
        # 没有任何分片，创建主文件
        return os.path.join(h5_dir, f"{base_h5_name}.h5")
    
    # 获取最后一个分片的编号和数据集数量
    last_part_num, last_path = part_files[-1]
    with h5py.File(last_path, 'r') as f:
        count = len(f.keys())
    
    if count >= MAX_DATASETS_PER_FILE:
        # 当前文件已满，创建下一个分片
        next_num = last_part_num + 1
        new_path = os.path.join(h5_dir, f"{base_h5_name}_part{next_num}.h5")
        print(f"当前分片 {last_path} 已满 ({count} 个数据集)，将创建新分片：{new_path}")
        return new_path
    else:
        print(f"继续使用当前分片：{last_path}（当前 {count} 个数据集）")
        return last_path

# ===== 主程序 =====
print("读取 FASTA...")
seq_ids = [record.id for record in SeqIO.parse(fasta_path, "fasta")]
total = len(seq_ids)
print(f"共 {total} 条序列。")

# 获取已有分片文件
part_files = get_existing_part_files()
print(f"发现 {len(part_files)} 个已有分片文件。")
for num, path in part_files:
    with h5py.File(path, 'r') as f:
        cnt = len(f.keys())
    print(f"  part{num}: {path} 包含 {cnt} 个数据集")

# 收集所有已有的 keys
existing_keys = get_all_existing_keys(part_files)
print(f"已有总数据集数量：{len(existing_keys)}")

to_process = [sid for sid in seq_ids if sid not in existing_keys]
print(f"还需处理 {len(to_process)} 个序列。")
if not to_process:
    print("✅ 全部完成。")
    exit(0)

# 决定输出文件
output_path = get_output_file(part_files)
print(f"本次将写入：{output_path}")

# ===== 打开输出文件（追加模式） =====
with h5py.File(output_path, 'a') as hf:
    # 再次检查该文件内是否已有部分数据（避免中断后重复）
    already_in_this_file = set(hf.keys())
    to_process_now = [sid for sid in to_process if sid not in already_in_this_file]
    if len(to_process_now) < len(to_process):
        print(f"当前文件内已有 {len(to_process) - len(to_process_now)} 个，实际将写入 {len(to_process_now)} 个。")
    if not to_process_now:
        print("✅ 当前文件已包含所有待处理序列。")
        exit(0)

    # 定义读取单个 .npy 的任务
    def read_npy(seq_id):
        npy_path = os.path.join(npy_dir, f"seq_{seq_id}.npy")
        if not os.path.exists(npy_path):
            return seq_id, None
        try:
            mat = np.load(npy_path)
            return seq_id, mat
        except Exception as e:
            print(f"⚠️ 读取 {seq_id} 失败: {e}")
            return seq_id, None

    # 使用队列缓存结果
    result_queue = queue.Queue(maxsize=100)

    def producer():
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(read_npy, sid): sid for sid in to_process_now}
            for future in as_completed(futures):
                sid, mat = future.result()
                result_queue.put((sid, mat))
        result_queue.put((None, None))  # 结束标志

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # 主线程消费队列并写入 HDF5
    processed = 0
    with tqdm(total=len(to_process_now), desc="写入 HDF5") as pbar:
        while True:
            sid, mat = result_queue.get()
            if sid is None and mat is None:
                break
            if mat is not None:
                # 保持 gzip 压缩（可根据需要改为 lzf 或 None）
                hf.create_dataset(sid, data=mat, compression="gzip")
            else:
                print(f"⚠️ 跳过 {sid}（文件缺失或读取失败）")
            processed += 1
            pbar.update(1)

    producer_thread.join()
    print(f"✅ 完成！本次共写入 {processed} 个序列。")
    print(f"文件 {output_path} 现有总数据集 {len(hf.keys())}。")