"""
app.py — Flask 服务器
将 utils.py 的哈夫曼算法以 HTTP API 的形式暴露给前端 HTML。

路由：
    GET  /              → 返回 index.html 页面
    POST /api/analyze   → 接收文本，返回频率表、编码表、压缩统计

运行方式（在 lab01/ 目录下）：
    python src/app.py
然后浏览器访问 http://127.0.0.1:5000
"""

import sys
import os
import chardet

def read_text(path: str) -> str:
    raw = open(path, 'rb').read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc)
# 保证 utils.py 可被直接 import（无论从哪个目录启动）
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from utils import (
    build_frequency_table,
    build_huffman_tree,
    generate_codes,
)

# templates/ 文件夹在 lab01/ 根目录，Flask 默认从 app.py 同级找，
# 所以需要手动指定路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # lab01/
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)


# ── 主页 ──────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── 分析接口 ──────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    请求体（JSON）：
        { "text": "<文件内容字符串>" }

    响应（JSON）：
        {
          "total":      总字符数,
          "unique":     不同字符种数,
          "orig_bytes": 原始 UTF-8 字节数,
          "comp_bytes": 哈夫曼压缩后字节数（不含文件头）,
          "ratio":      压缩率（百分比，保留1位小数）,
          "avg_bits":   平均编码长度（bit/字符）,
          "freq":       { 字符: 频率, ... },
          "codes":      { 字符: "010...", ... }
        }
    """
    data = request.get_json(force=True)
    text: str = data.get("text", "")

    if not text:
        return jsonify({"error": "文本不能为空"}), 400

    try:
        # ── 调用 utils.py 的核心算法 ──
        freq_table = build_frequency_table(text)
        root       = build_huffman_tree(freq_table)
        codes      = generate_codes(root)

        # ── 统计 ──
        orig_bytes = len(text.encode("utf-8"))
        total_bits = sum(freq_table[ch] * len(codes[ch]) for ch in freq_table)
        comp_bytes = (total_bits + 7) // 8          # 向上取整
        ratio      = (1 - comp_bytes / orig_bytes) * 100
        avg_bits   = total_bits / len(text)

        return jsonify({
            "total":      len(text),
            "unique":     len(freq_table),
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio":      round(ratio, 1),
            "avg_bits":   round(avg_bits, 2),
            "freq":       freq_table,
            "codes":      codes,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ── 启动 ──────────────────────────────────────
if __name__ == "__main__":
    print("启动哈夫曼可视化服务器...")
    print("请用浏览器访问 → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)