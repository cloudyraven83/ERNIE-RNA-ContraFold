#!/usr/bin/env python3
import os
import glob
import argparse

def parse_ct_file(filepath):
    """
    解析CT文件，返回配对集合（每个配对为排序后的 (i, j)，i<j）
    CT文件格式：每行包含索引、碱基、前索引、后索引、配对索引、自然索引
    配对索引>0表示有配对
    """
    pairs = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return pairs
    # 跳过第一行（通常是长度和注释）
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            pair_idx = int(parts[4])
            if pair_idx > 0:
                i = idx
                j = pair_idx
                if i != j:
                    if i > j:
                        i, j = j, i
                    pairs.add((i, j))
        except ValueError:
            continue
    return pairs

def compute_f1(pred_pairs, true_pairs):
    """计算精确度、召回率、F1，并返回TP, FP, FN"""
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn

def main():
    parser = argparse.ArgumentParser(description='计算RNA二级结构预测的F1分数')
    parser.add_argument('--true_dir', default='~/ERNIE-RNA/Ltwo/ctFiles/',
                        help='真实CT文件目录')
    parser.add_argument('--pred_dir', default='~/ERNIE-RNA/Ltwo/results/zeroshot_prediction/',
                        help='预测CT文件目录')
    parser.add_argument('--output', default='f1_results.txt',
                        help='输出结果文件')
    args = parser.parse_args()

    true_dir = os.path.expanduser(args.true_dir)
    pred_dir = os.path.expanduser(args.pred_dir)

    # 获取所有预测文件（_zeroshot_prediction.ct）
    pred_pattern = os.path.join(pred_dir, '*_zeroshot_prediction.ct')
    pred_files = glob.glob(pred_pattern)
    # 提取 seq_id（去掉 _zeroshot_prediction.ct 后缀）
    ids = [os.path.basename(f).replace('_zeroshot_prediction.ct', '') for f in pred_files]

    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for seq_id in ids:
        true_file = os.path.join(true_dir, f'{seq_id}.ct')
        pred_file = os.path.join(pred_dir, f'{seq_id}_zeroshot_prediction.ct')
        if not os.path.exists(true_file):
            print(f"警告: 真实文件 {true_file} 不存在，跳过")
            continue
        true_pairs = parse_ct_file(true_file)
        pred_pairs = parse_ct_file(pred_file)
        prec, rec, f1, tp, fp, fn = compute_f1(pred_pairs, true_pairs)
        results.append((seq_id, prec, rec, f1, tp, fp, fn))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 计算宏平均和微平均
    num = len(results)
    if num > 0:
        avg_prec = sum(r[1] for r in results) / num
        avg_rec = sum(r[2] for r in results) / num
        avg_f1 = sum(r[3] for r in results) / num
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    else:
        avg_prec = avg_rec = avg_f1 = 0.0
        micro_prec = micro_rec = micro_f1 = 0.0

    # 写入输出文件
    with open(args.output, 'w') as out:
        out.write("seq_id\tprecision\trecall\tf1\ttp\tfp\tfn\n")
        for seq_id, prec, rec, f1, tp, fp, fn in results:
            out.write(f"{seq_id}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\n")
        out.write("\n=== Summary ===\n")
        out.write(f"Number of sequences: {num}\n")
        out.write(f"Macro-average precision: {avg_prec:.4f}\n")
        out.write(f"Macro-average recall: {avg_rec:.4f}\n")
        out.write(f"Macro-average F1: {avg_f1:.4f}\n")
        out.write(f"Micro-average precision: {micro_prec:.4f}\n")
        out.write(f"Micro-average recall: {micro_rec:.4f}\n")
        out.write(f"Micro-average F1: {micro_f1:.4f}\n")
        out.write(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}\n")

    print(f"结果已写入 {args.output}")

if __name__ == '__main__':
    main()