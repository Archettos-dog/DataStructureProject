"""
test_main.py — 哈夫曼编码各模块单元测试
运行方式：
    pytest tests/test_main.py -v
    python -m pytest tests/test_main.py -v   # 从项目根目录
"""

import os
import sys
import tempfile
import pytest


# 将 src/ 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    HuffmanNode,
    build_frequency_table,
    build_huffman_tree,
    generate_codes,
    compress,
    decompress,
    _text_to_bitstring,
    _bitstring_to_bytes,
    _bytes_to_bitstring,
    decode_bitstring,
)


# ══════════════════════════════════════════════
# 1. 频率统计
# ══════════════════════════════════════════════

class TestBuildFrequencyTable:
    def test_basic_english(self):
        freq = build_frequency_table("aab")
        assert freq == {"a": 2, "b": 1}

    def test_chinese(self):
        freq = build_frequency_table("你好你")
        assert freq == {"你": 2, "好": 1}

    def test_mixed(self):
        freq = build_frequency_table("abc，。！")
        assert "，" in freq and "。" in freq and "！" in freq

    def test_full_and_half_width(self):
        # 全角逗号（，）与半角逗号（,）应被识别为不同字符
        freq = build_frequency_table("，,")
        assert freq.get("，") == 1
        assert freq.get(",") == 1

    def test_single_char(self):
        freq = build_frequency_table("aaaa")
        assert freq == {"a": 4}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_frequency_table("")

    def test_space_and_newline(self):
        freq = build_frequency_table("a b\nc")
        assert " " in freq
        assert "\n" in freq


# ══════════════════════════════════════════════
# 2. 哈夫曼树构建
# ══════════════════════════════════════════════

class TestBuildHuffmanTree:
    def test_root_freq_equals_total(self):
        freq = {"a": 5, "b": 3, "c": 2}
        root = build_huffman_tree(freq)
        assert root.freq == 10

    def test_single_char(self):
        # 只有一种字符：根有一个左孩子，该孩子是叶节点
        root = build_huffman_tree({"x": 7})
        assert root.char is None
        assert root.left is not None
        assert root.left.char == "x"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_huffman_tree({})

    def test_all_leaves_are_chars(self):
        freq = {"a": 1, "b": 2, "c": 3, "d": 4}
        root = build_huffman_tree(freq)

        leaves = []
        def collect(node):
            if node is None:
                return
            if node.char is not None:
                leaves.append(node.char)
            collect(node.left)
            collect(node.right)

        collect(root)
        assert sorted(leaves) == ["a", "b", "c", "d"]


# ══════════════════════════════════════════════
# 3. 编码表生成
# ══════════════════════════════════════════════

class TestGenerateCodes:
    def test_codes_are_prefix_free(self):
        """哈夫曼编码必须满足前缀码性质"""
        freq = {"a": 5, "b": 3, "c": 2, "d": 1}
        root = build_huffman_tree(freq)
        codes = generate_codes(root)

        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    assert not c2.startswith(c1), f"{c1!r} 是 {c2!r} 的前缀，违反前缀码"

    def test_high_freq_shorter(self):
        """高频字符的编码长度 ≤ 低频字符"""
        freq = {"a": 100, "b": 5}
        root = build_huffman_tree(freq)
        codes = generate_codes(root)
        assert len(codes["a"]) <= len(codes["b"])

    def test_single_char_code(self):
        root = build_huffman_tree({"z": 1})
        codes = generate_codes(root)
        assert codes["z"] in ("0", "1")

    def test_all_chars_have_code(self):
        freq = {"a": 1, "b": 2, "c": 3}
        root = build_huffman_tree(freq)
        codes = generate_codes(root)
        assert set(codes.keys()) == {"a", "b", "c"}


# ══════════════════════════════════════════════
# 4. 比特流转换
# ══════════════════════════════════════════════

class TestBitConversion:
    def test_roundtrip_aligned(self):
        """恰好8位倍数的比特串，padding 应为 0"""
        bits = "10110010"
        data, padding = _bitstring_to_bytes(bits)
        assert padding == 0
        assert _bytes_to_bitstring(data, padding) == bits

    def test_roundtrip_unaligned(self):
        """不足8位的比特串，补位后能正确还原"""
        bits = "101"
        data, padding = _bitstring_to_bytes(bits)
        assert padding == 5
        assert _bytes_to_bitstring(data, padding) == bits

    def test_leading_zeros_preserved(self):
        """字节值小（如 0b00000011）时前导零必须保留"""
        bits = "00000011"
        data, padding = _bitstring_to_bytes(bits)
        assert _bytes_to_bitstring(data, padding) == bits

    def test_padding_zero_edge_case(self):
        """比特串长度恰好是8的倍数，padding 不能错误地补8位"""
        bits = "1" * 16
        data, padding = _bitstring_to_bytes(bits)
        assert padding == 0
        assert len(data) == 2


# ══════════════════════════════════════════════
# 5. 解码
# ══════════════════════════════════════════════

class TestDecoding:
    def _setup(self, text: str):
        freq = build_frequency_table(text)
        root = build_huffman_tree(freq)
        codes = generate_codes(root)
        return root, codes

    def test_decode_simple(self):
        text = "abac"
        root, codes = self._setup(text)
        bits = _text_to_bitstring(text, codes)
        assert decode_bitstring(bits, root) == text

    def test_decode_chinese(self):
        text = "哈夫曼编码"
        root, codes = self._setup(text)
        bits = _text_to_bitstring(text, codes)
        assert decode_bitstring(bits, root) == text

    def test_decode_single_char_repeated(self):
        text = "aaaa"
        root, codes = self._setup(text)
        bits = _text_to_bitstring(text, codes)
        assert decode_bitstring(bits, root) == text


# ══════════════════════════════════════════════
# 6. 压缩与解压（端到端测试）
# ══════════════════════════════════════════════

class TestCompressDecompress:

    def _roundtrip(self, text: str) -> None:
        """辅助方法：写临时文件 → 压缩 → 解压 → 对比内容"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.txt")
            huf = os.path.join(tmpdir, "output.huf")
            dst = os.path.join(tmpdir, "restored.txt")

            with open(src, "w", encoding="utf-8") as f:
                f.write(text)

            compress(src, huf)
            decompress(huf, dst)

            with open(dst, "r", encoding="utf-8") as f:
                restored = f.read()

        assert restored == text, "解压内容与原文不一致"

    def test_english_text(self):
        self._roundtrip("the quick brown fox jumps over the lazy dog.")

    def test_chinese_text(self):
        self._roundtrip("哈夫曼编码是一种贪心算法，用于无损数据压缩。")

    def test_mixed_text(self):
        self._roundtrip("Hello 世界！Huffman(哈夫曼) coding，压缩率约 40%。")

    def test_single_char(self):
        self._roundtrip("a" * 100)

    def test_with_spaces_and_newlines(self):
        self._roundtrip("line one\nline two\nline three\n")

    def test_full_width_punctuation(self):
        self._roundtrip("你好，世界！这是全角标点。")

    def test_long_text(self):
        # 较长文本，验证多字节边界处理正确
        text = "数据结构与算法 " * 200 + "end."
        self._roundtrip(text)

    def test_compressed_smaller_than_original(self):
        """重复性高的文本，压缩后应比原文小"""
        text = "abcabc" * 500
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.txt")
            huf = os.path.join(tmpdir, "output.huf")
            with open(src, "w", encoding="utf-8") as f:
                f.write(text)
            stats = compress(src, huf)
        assert stats["compressed_size"] < stats["original_size"]

    def test_stats_fields(self):
        """compress() 返回的统计字典应包含所有必要字段"""
        text = "hello world"
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.txt")
            huf = os.path.join(tmpdir, "output.huf")
            with open(src, "w", encoding="utf-8") as f:
                f.write(text)
            stats = compress(src, huf)

        for key in ("original_size", "compressed_size", "ratio",
                    "char_count", "unique_chars", "avg_code_len", "codes", "freq_table"):
            assert key in stats, f"统计字典缺少字段：{key}"

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "empty.txt")
            huf = os.path.join(tmpdir, "output.huf")
            open(src, "w").close()
            with pytest.raises(ValueError):
                compress(src, huf)