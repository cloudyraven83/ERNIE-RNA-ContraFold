#!/usr/bin/env python3
import os
from Bio import SeqIO

def load_allowed_chars(dict_path):
    """从 dict.txt 中提取合法的核苷酸符号（忽略特殊占位词）"""
    allowed = set()
    with open(dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[0]
            if token.startswith('madeupword') or token.startswith('<'):
                continue
            if token.isupper() and len(token) == 1:
                allowed.add(token)
    return allowed

def clean_fasta_to_txt(fasta_in, txt_out, dict_path, min_len=10, max_len=1024):
    """清洗单条 FASTA 文件，输出空格分隔的序列文本"""
    allowed_chars = load_allowed_chars(dict_path)
    print(f"允许的碱基字符: {sorted(allowed_chars)}")

    total = 0
    kept = 0
    skipped_reason = {'invalid_char': 0, 'length': 0}

    with open(txt_out, 'w') as f_out:
        for record in SeqIO.parse(fasta_in, "fasta"):
            total += 1
            seq = str(record.seq).upper()
            seq = seq.replace('T', 'U')   # 替换 T -> U

            if not set(seq).issubset(allowed_chars):
                skipped_reason['invalid_char'] += 1
                continue

            seq_len = len(seq)
            if not (min_len <= seq_len <= max_len):
                skipped_reason['length'] += 1
                continue

            # 输出空格分隔的字符（与 fairseq 的 tokenization 兼容）
            f_out.write(' '.join(seq) + '\n')
            kept += 1

    print(f"处理完成！")
    print(f"总序列数: {total}")
    print(f"保留序列数: {kept}")
    print(f"跳过原因: 含非法字符 = {skipped_reason['invalid_char']}, 长度不符 = {skipped_reason['length']}")
    print(f"清洗后文件保存在: {txt_out}")

if __name__ == "__main__":
    dict_path = os.path.expanduser("~/ERNIE-RNA/src/dict/dict.txt")
    base_data_dir = os.path.expanduser("~/ERNIE-RNA/data-bin")

    # 三个分片的输入 FASTA 和输出 TXT 路径
    splits = {
        "train": {
            "fasta_in": os.path.join(base_data_dir, "train", "train.fasta"),
            "txt_out": os.path.join(base_data_dir, "train", "train_cleaned.txt"),
        },
        "valid": {
            "fasta_in": os.path.join(base_data_dir, "valid", "valid.fasta"),
            "txt_out": os.path.join(base_data_dir, "valid", "valid_cleaned.txt"),
        },
        "test": {
            "fasta_in": os.path.join(base_data_dir, "test", "test.fasta"),
            "txt_out": os.path.join(base_data_dir, "test", "test_cleaned.txt"),
        },
    }

    if not os.path.exists(dict_path):
        print(f"错误：找不到词典文件 {dict_path}")
        exit(1)

    for split_name, paths in splits.items():
        if not os.path.exists(paths["fasta_in"]):
            print(f"警告：{split_name} 的输入文件不存在，跳过")
            continue
        print(f"\n===== 处理 {split_name} 分片 =====")
        clean_fasta_to_txt(
            fasta_in=paths["fasta_in"],
            txt_out=paths["txt_out"],
            dict_path=dict_path,
            min_len=10,
            max_len=1024
        )