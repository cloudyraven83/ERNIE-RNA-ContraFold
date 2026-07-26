#!/usr/bin/env python3
"""
评估ERNIE-RNA所有层-头组合在RNA二级结构预测上的性能
python evaluate_heads.py \
    --fasta_dir ~/ERNIE-RNA/Ltwo/fastaFiles \
    --ct_dir ~/ERNIE-RNA/Ltwo/ctFiles \
    --contact_h5 /path/to/real_contact_maps.h5 \
    --device 0
"""

import os
import re
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ========== 进度条支持（自动降级） ==========
try:
    from tqdm import tqdm
except ImportError:
    # 简易替代（无依赖）
    class tqdm:
        def __init__(self, iterable, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total}", end='', flush=True)
                else:
                    print(f"\r{self.desc}: {self.n}", end='', flush=True)
            if self.total:
                print()
        def __enter__(self):
            return self
        def __exit__(self, *args, **kwargs):
            pass

# ========== 导入ERNIE-RNA组件 ==========
from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import load_pretrained_ernierna, prepare_input_for_ernierna, read_fasta_file

# ==================== 基础辅助函数 ====================

def seq_to_rnaindex_and_onehot(seq):
    l = len(seq)
    X = np.ones((1, l+2))
    data_seq = torch.zeros((1, l, 4))
    for j in range(l):
        if seq[j] in set('Aa'):
            X[0, j+1] = 5
            data_seq[0, j] = torch.Tensor([1,0,0,0])
        elif seq[j] in set('UuTt'):
            X[0, j+1] = 6
            data_seq[0, j] = torch.Tensor([0,1,0,0])
        elif seq[j] in set('Cc'):
            X[0, j+1] = 7
            data_seq[0, j] = torch.Tensor([0,0,1,0])
        elif seq[j] in set('Gg'):
            X[0, j+1] = 4
            data_seq[0, j] = torch.Tensor([0,0,0,1])
        else:
            X[0, j+1] = 3
            data_seq[0, j] = torch.Tensor([0,0,0,0])
    X[0, l+1] = 2
    X[0, 0] = 0
    return X, data_seq

def constraint_matrix_batch(x):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu

def soft_sign(x):
    return 1.0/(1.0+torch.exp(-2*x))

def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def post_process(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=None):
    m = constraint_matrix_batch(x).float()
    a_hat = torch.sigmoid(u)
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()
    for t in range(num_itr):
        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99
        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)
        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

# ==================== 评估相关函数 ====================

def parse_ct_file(ct_path):
    """
    解析CT文件，返回真实碱基对集合 (0-based)
    跳过注释行（以#开头）和空行
    """
    pairs = set()
    with open(ct_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return pairs
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0]) - 1
                pair_idx = int(parts[4]) - 1
            except ValueError:
                # 如果某行格式不对，跳过
                continue
            if pair_idx >= 0:
                pairs.add((min(idx, pair_idx), max(idx, pair_idx)))
    return pairs

def contact_matrix_to_pairs(matrix, threshold=0.5):
    pairs = set()
    L = matrix.shape[0]
    for i in range(L):
        for j in range(i+1, L):
            if matrix[i, j] > threshold:
                pairs.add((i, j))
    return pairs

def calculate_f1(pred_pairs, true_pairs):
    if not true_pairs:
        return 0.0
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ==================== 主评估类 ====================

class ErnieRNAWrapper(nn.Module):
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder
    def forward(self, x, twod_input):
        _, attn_map, _ = self.sentence_encoder(x, twod_tokens=twod_input,
                                               is_twod=True, extra_only=True,
                                               masked_only=False)
        return attn_map

def evaluate_all_heads(seqs_dict, ct_dir, pretrain_model, device,
                       contact_h5=None, threshold=0.5):
    """
    评估所有层-头组合，返回平均F1矩阵 (12,12)
    增加进度条显示
    """
    total_f1 = np.zeros((12, 12))
    count = np.zeros((12, 12))
    num_seqs = len(seqs_dict)

    # 使用进度条包装迭代
    for seq_name, seq in tqdm(seqs_dict.items(), total=num_seqs, desc="Evaluating sequences"):
        seq_id = seq_name.split()[0].split('|')[0]
        safe_seq_id = re.sub(r'[^a-zA-Z0-9_]', '_', seq_id)
        ct_path = os.path.join(ct_dir, f"{safe_seq_id}.ct")
        if not os.path.exists(ct_path):
            # 静默跳过（避免过多警告）
            continue

        # 准备输入
        X, data_seq = seq_to_rnaindex_and_onehot(seq)
        one_d, twod_data = prepare_input_for_ernierna(X, len(seq),
                                                      contact_h5=contact_h5,
                                                      seq_id=seq_id)
        oned = one_d.to(device)
        twod_data = twod_data.to(device)
        data_seq = data_seq.to(device)

        with torch.no_grad():
            attn_maps = pretrain_model(oned, twod_data)  # list of 12 tensors
            attn_all = torch.stack(attn_maps)  # (12, 1, 12, T, T)

        true_pairs = parse_ct_file(ct_path)

        # 遍历所有层和头
        for l in range(12):
            for h in range(12):
                attn = attn_all[l, 0, h, :, :]
                attn_no_special = attn[1:-1, 1:-1]
                pair_attn = (attn_no_special + attn_no_special.T) / 2
                pair_attn_batch = pair_attn.unsqueeze(0)
                post_pair = post_process(pair_attn_batch, data_seq,
                                         0.01, 0.1, 100, 1.6, True, 1.5)
                contact = post_pair.squeeze().cpu().numpy()
                pred_pairs = contact_matrix_to_pairs(contact, threshold)
                f1 = calculate_f1(pred_pairs, true_pairs)
                total_f1[l, h] += f1
                count[l, h] += 1

    avg_f1 = total_f1 / count
    return avg_f1

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="评估ERNIE-RNA所有层-头组合")
    parser.add_argument("--fasta_dir", type=str, default="~/ERNIE-RNA/Ltwo/fastaFiles",
                        help="验证集FASTA文件目录")
    parser.add_argument("--ct_dir", type=str, default="~/ERNIE-RNA/Ltwo/ctFiles",
                        help="真实CT文件目录")
    parser.add_argument("--contact_h5", type=str, default=None,
                        help="预计算接触矩阵HDF5文件路径（可选）")
    parser.add_argument("--arg_overrides", type=str, default='{"data": "./src/dict/"}',
                        help="字典路径参数")
    parser.add_argument("--ernie_rna_checkpoint", type=str,
                        default="./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt",
                        help="预训练模型路径")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU设备号，若为-1则使用CPU")
    parser.add_argument("--output_dir", type=str, default="./head_evaluation/",
                        help="结果保存目录")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="接触矩阵二值化阈值")
    args = parser.parse_args()

    # 扩展路径
    fasta_dir = os.path.expanduser(args.fasta_dir)
    ct_dir = os.path.expanduser(args.ct_dir)
    contact_h5 = os.path.expanduser(args.contact_h5) if args.contact_h5 else None

    # 检查contact_h5文件是否存在，若不存在则置为None并提示
    if contact_h5 is not None and not os.path.isfile(contact_h5):
        print(f"Warning: Contact HDF5 file '{contact_h5}' not found. Will use base-pairing rules (creatmat).")
        contact_h5 = None

    # ===== 读取所有FASTA文件（带进度显示） =====
    print(f"Scanning FASTA directory: {fasta_dir}")
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(('.fasta', '.fa'))]
    print(f"Found {len(fasta_files)} FASTA file(s).")
    
    all_seqs = {}
    for idx, fname in enumerate(fasta_files, 1):
        print(f"Loading file {idx}/{len(fasta_files)}: {fname}", flush=True)
        filepath = os.path.join(fasta_dir, fname)
        seqs_dict = read_fasta_file(filepath)
        all_seqs.update(seqs_dict)
        print(f"  -> loaded {len(seqs_dict)} sequences (total now: {len(all_seqs)})", flush=True)
    
    print(f"Total sequences loaded: {len(all_seqs)}")

    # 设备设置
    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device: {device}")

    # 加载预训练模型
    import json
    arg_overrides = json.loads(args.arg_overrides) if isinstance(args.arg_overrides, str) else args.arg_overrides
    print("Loading pre-trained model...")
    model_pre = load_pretrained_ernierna(args.ernie_rna_checkpoint, arg_overrides)
    pretrain_model = ErnieRNAWrapper(model_pre.encoder).to(device)
    pretrain_model.eval()
    print("Model loaded.")

    # 执行评估
    start_time = time.time()
    avg_f1 = evaluate_all_heads(all_seqs, ct_dir, pretrain_model, device,
                                contact_h5=contact_h5,
                                threshold=args.threshold)
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds.")

    # 找出最佳
    best_layer, best_head = np.unravel_index(np.argmax(avg_f1), avg_f1.shape)
    best_f1 = avg_f1[best_layer, best_head]

    # 输出结果
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'avg_f1_matrix.npy'), avg_f1)

    with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as f:
        f.write(f"Best layer: {best_layer}\n")
        f.write(f"Best head: {best_head}\n")
        f.write(f"Best average F1: {best_f1:.6f}\n")
        f.write("\nFull F1 matrix (12x12):\n")
        for l in range(12):
            f.write("  ".join(f"{avg_f1[l, h]:.4f}" for h in range(12)) + "\n")

    print(f"\n=== Best Configuration ===")
    print(f"Layer: {best_layer}, Head: {best_head}")
    print(f"Average F1: {best_f1:.4f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()