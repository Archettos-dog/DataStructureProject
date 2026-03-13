# save as inspect_unicode.py and run: python inspect_unicode.py path/to/your.txt
import sys, unicodedata

def inspect_file(path, limit=500):
    with open(path, 'r', encoding='utf-8', errors='surrogatepass') as f:
        text = f.read()

    # 统计频率（按 codepoint）
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # 找出频次小但可能“不可见”的字符（例如频次<=10 的项）
    candidates = [ (ch, cnt) for ch,cnt in freq.items() if cnt <= 50 ]  # 可调整阈值
    candidates.sort(key=lambda x: (-x[1], ord(x[0])))

    def info(ch):
        cp = ord(ch)
        hexcp = f"U+{cp:04X}"
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "<unassigned or no name>"
        cat = unicodedata.category(ch)  # e.g., Cc Cf Mn Lo etc
        utf8_bytes = ch.encode('utf-8', 'backslashreplace')
        return hexcp, name, cat, utf8_bytes

    print(f"Total unique codepoints: {len(freq)}\n")
    print("Sample report for candidates (char | count | codepoint | name | category | utf8-bytes | repr):\n")
    for ch, cnt in candidates[:200]:
        hexcp, name, cat, utf8 = info(ch)
        # show readable repr (escape control + show replacements)
        printable = ch if (not ch.isspace() and unicodedata.category(ch)[0] != 'C') else repr(ch)
        print(f"{printable} | {cnt} | {hexcp} | {name} | {cat} | {utf8} | repr={repr(ch)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_unicode.py path/to/file.txt")
    else:
        inspect_file(sys.argv[1])