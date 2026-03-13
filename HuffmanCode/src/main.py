"""
main.py — 哈夫曼编码压缩工具入口
用法：
    python main.py compress <input.txt> [output.huf]
    python main.py decompress <input.huf> [output.txt]
    python main.py info <input.txt>          # 只查看频率和编码表，不压缩
"""

import sys
import os
import argparse

# 将 src/ 加入模块搜索路径，支持从项目根目录运行
sys.path.insert(0, os.path.dirname(__file__))

from utils import compress, decompress, print_code_table
from utils import build_frequency_table, build_huffman_tree, generate_codes


# ─────────────────────────────────────────────
# 输出格式化工具
# ─────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    """将字节数格式化为人类可读的字符串。"""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"
    else:
        return f"{n / 1024 ** 2:.2f} MB"


def _print_header(title: str) -> None:
    print("\n" + "═" * 56)
    print(f"  {title}")
    print("═" * 56)


def _print_kv(key: str, value: str, width: int = 20) -> None:
    print(f"  {key:<{width}} {value}")


# ─────────────────────────────────────────────
# 子命令：info（查看频率和编码表）
# ─────────────────────────────────────────────

def cmd_info(input_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[错误] 文件不存在：{input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text:
        print("[错误] 文件内容为空")
        sys.exit(1)

    freq_table = build_frequency_table(text)
    root = build_huffman_tree(freq_table)
    codes = generate_codes(root)

    _print_header("文件信息与哈夫曼编码表")
    _print_kv("文件路径", input_path)
    _print_kv("文件大小", _fmt_size(os.path.getsize(input_path)))
    _print_kv("总字符数", str(len(text)))
    _print_kv("不同字符种数", str(len(freq_table)))

    total_bits = sum(freq * len(codes[ch]) for ch, freq in freq_table.items())
    _print_kv("平均编码长度", f"{total_bits / len(text):.4f} bit/字符")

    print_code_table(codes, freq_table)


# ─────────────────────────────────────────────
# 子命令：compress（压缩）
# ─────────────────────────────────────────────

def cmd_compress(input_path: str, output_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[错误] 文件不存在：{input_path}")
        sys.exit(1)

    print(f"\n正在压缩：{input_path} → {output_path}")

    stats = compress(input_path, output_path)

    _print_header("压缩完成")
    _print_kv("原始文件", input_path)
    _print_kv("压缩文件", output_path)
    _print_kv("原始大小", _fmt_size(stats["original_size"]))
    _print_kv("压缩后大小", _fmt_size(stats["compressed_size"]))
    _print_kv("压缩率", f"{stats['ratio']:.2f}%")
    _print_kv("总字符数", str(stats["char_count"]))
    _print_kv("不同字符种数", str(stats["unique_chars"]))
    _print_kv("平均编码长度", f"{stats['avg_code_len']:.4f} bit/字符")

    # 打印编码表
    print_code_table(stats["codes"], stats["freq_table"])


# ─────────────────────────────────────────────
# 子命令：decompress（解压）
# ─────────────────────────────────────────────

def cmd_decompress(input_path: str, output_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[错误] 文件不存在：{input_path}")
        sys.exit(1)

    print(f"\n正在解压：{input_path} → {output_path}")

    stats = decompress(input_path, output_path)

    _print_header("解压完成")
    _print_kv("压缩文件", input_path)
    _print_kv("还原文件", output_path)
    _print_kv("压缩文件大小", _fmt_size(stats["compressed_size"]))
    _print_kv("还原文件大小", _fmt_size(stats["decompressed_size"]))
    _print_kv("还原字符数", str(stats["char_count"]))

    print("\n[提示] 可用 diff 命令验证还原是否完全一致：")
    print(f"  diff <原始文件> {output_path}")


# ─────────────────────────────────────────────
# 参数解析与入口
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huffman",
        description="哈夫曼编码文件压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python main.py compress  article.txt
  python main.py compress  article.txt archive.huf
  python main.py decompress archive.huf restored.txt
  python main.py info      article.txt
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="压缩文本文件")
    p_compress.add_argument("input", help="输入 .txt 文件路径")
    p_compress.add_argument(
        "output", nargs="?", help="输出 .huf 文件路径（默认与输入同名）"
    )

    # decompress
    p_decompress = sub.add_parser("decompress", help="解压 .huf 文件")
    p_decompress.add_argument("input", help="输入 .huf 文件路径")
    p_decompress.add_argument(
        "output", nargs="?", help="输出 .txt 文件路径（默认与输入同名加 _restored）"
    )

    # info
    p_info = sub.add_parser("info", help="查看文件字符频率和哈夫曼编码表")
    p_info.add_argument("input", help="输入 .txt 文件路径")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args.input)

    elif args.command == "compress":
        output = args.output or os.path.splitext(args.input)[0] + ".huf"
        cmd_compress(args.input, output)

    elif args.command == "decompress":
        base = os.path.splitext(args.input)[0]
        output = args.output or base + "_restored.txt"
        cmd_decompress(args.input, output)


if __name__ == "__main__":
    main()