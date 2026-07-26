#!/usr/bin/env python3
import sys
import os
import time

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_fasta_with_log.py <input.fasta> <output.fasta>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在：{input_file}")
        sys.exit(1)

    print(f"开始处理：{input_file}")
    print(f"输出到：{output_file}")

    total_lines = 0
    seq_lines = 0
    start_time = time.time()

    try:
        with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
            for line in inf:
                total_lines += 1
                if line.startswith('>'):
                    outf.write(line)
                else:
                    seq_lines += 1
                    # 转大写
                    line = line.upper()
                    # T -> U
                    line = line.replace('T', 'U')
                    # 写入（保留原换行符）
                    outf.write(line)

                # 每处理 100 万行打印一次进度
                if total_lines % 1000000 == 0:
                    elapsed = time.time() - start_time
                    print(f"已处理 {total_lines} 行（其中序列行 {seq_lines} 行），用时 {elapsed:.1f} 秒")

        elapsed = time.time() - start_time
        print(f"处理完成！总行数：{total_lines}，序列行数：{seq_lines}，总用时：{elapsed:.1f} 秒")
        print(f"输出文件已保存至：{output_file}")

        # 检查输出文件是否生成
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"输出文件大小：{size} 字节（{size/1024/1024:.2f} MB）")
        else:
            print("警告：输出文件似乎未被创建！")

    except Exception as e:
        print(f"处理过程中出现错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()