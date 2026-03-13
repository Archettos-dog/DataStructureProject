# generate_test.py
# 生成用于哈夫曼编码健壮性测试的文本文件（UTF-8 编码，包含各种字符及控制字节）

with open('test_huffman.txt', 'wb') as f:
    # 可选写入 UTF-8 BOM（可测试程序是否忽略或正确处理）
    f.write(b'\xef\xbb\xbf')
    f.write(b'=== Huffman Coding Robustness Test File ===\n')
    f.write(b'This file contains a variety of characters to test your Huffman compression program.\n\n')

    # 1. ASCII 可打印字符（全部 95 个）
    f.write(b'1. ASCII printable:\n')
    f.write(b' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n')

    # 2. ASCII 控制字符（原始字节，包含 NUL、BEL、ESC、DEL 等）
    f.write(b'2. ASCII control chars (raw bytes):\n')
    # 写入 0x00-0x1F 以及 0x7F
    f.write(bytes(range(0x20)) + b'\x7f\n')

    # 3. Latin-1 扩展字符（UTF-8 编码）
    f.write(b'3. Latin-1 chars: \xc3\xa9 \xc3\xa8 \xc3\xb1 \xc3\xbc \xc3\xbf\n')  # é è ñ ü ÿ

    # 4. 中文、全角标点、生僻汉字
    f.write('4. Chinese: 中文测试，全角标点。！？“”【】等。生僻字: 𪚥 (U+2A6A5), 𠀀 (U+20000).\n'.encode('utf-8'))

    # 5. 日文、韩文、俄文、阿拉伯文
    f.write('5. Japanese: こんにちは (Konnichiwa)\n'.encode('utf-8'))
    f.write('6. Korean: 안녕하세요 (Annyeonghaseyo)\n'.encode('utf-8'))
    f.write('7. Russian: Привет (Privet)\n'.encode('utf-8'))
    f.write('8. Arabic: السلام عليكم (Al-salam alaykum)\n'.encode('utf-8'))

    # 6. 零宽空格及不可见控制符
    f.write('9. Zero-width space (U+200B): Zero\u200BWidth\u200BSpace\n'.encode('utf-8'))
    f.write('10. Zero-width non-joiner (U+200C): Zero\u200CWidth\u200C NonJoiner\n'.encode('utf-8'))
    f.write('11. Zero-width joiner (U+200D): Zero\u200DWidth\u200D Joiner\n'.encode('utf-8'))
    f.write('12. Left-to-right mark (U+200E): LTR\u200E Mark\n'.encode('utf-8'))
    f.write('13. Right-to-left mark (U+200F): RTL\u200F Mark\n'.encode('utf-8'))

    # 7. 组合字符
    f.write('14. Combining acute: e\u0301 (e+combining acute) -> é\n'.encode('utf-8'))
    f.write('15. Combining umlaut: a\u0308 o\u0308 u\u0308 -> ä ö ü\n'.encode('utf-8'))

    # 8. 表情符号
    f.write('16. Emojis: 😀 😁 😂 😃 😄 😅 😆 😇 😈 😉 😊 😋 😌 😍 😎 😏\n'.encode('utf-8'))

    # 9. 数学符号
    f.write('17. Math: ∑ ∫ ∂ ∇ ∞ ≈ ≠ ≤ ≥ ∏ √\n'.encode('utf-8'))

    # 10. 箭头
    f.write('18. Arrows: ← ↑ → ↓ ↔ ↕ ↖ ↗ ↘ ↙\n'.encode('utf-8'))

    # 11. 几何图形
    f.write('19. Shapes: ■ □ ▢ ▣ ▤ ▥ ▦ ▧ ▨ ▩ ▪ ▫\n'.encode('utf-8'))

    # 12. 音乐符号与货币
    f.write('20. Music: ♩ ♪ ♫ ♬\n'.encode('utf-8'))
    f.write('21. Currency: $ € £ ¥ ¢\n'.encode('utf-8'))

    # 13. 字节顺序标记（BOM）内嵌
    f.write('22. BOM inside: \uFEFF (zero-width no-break space)\n'.encode('utf-8'))

    # 14. 长重复序列
    f.write(b'23. Long run of A (100 times): ' + b'A'*100 + b'\n')
    f.write(b'24. Long run of nulls (20 bytes): ' + b'\x00'*20 + b' after nulls\n')

    # 15. 非常长且末尾无换行的行
    f.write(b'25. Very long line without newline (500 x\'s): ' + b'x'*500)
    # 下一行带换行，确保之前行无换行
    f.write(b'\n26. This line has a newline.\n')

    # 16. 所有字节值 0-255 混合（测试二进制安全）
    f.write(b'27. Mix of bytes 0-255 (all values):\n')
    f.write(bytes(range(256)))
    f.write(b'\n')

    # 17. 文件末尾无换行
    f.write(b'28. Final line without newline at end of file.')