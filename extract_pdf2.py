import re

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

text = content.decode('latin-1', errors='ignore')

# Check format
sample = text[:2000]
with open(r'd:\papers-code\Extremum_prevention\sample.txt', 'w', encoding='utf-8') as f:
    f.write(sample)

# Try different text extraction strategies
# Strategy 1: Look for text in streams
streams = re.findall(r'stream\n(.*?)endstream', text, re.DOTALL)
print(f'Found {len(streams)} streams')

# Strategy 2: Try Tj, TJ, ' operators
all_text = []
for s in streams:
    # Tj operator
    tjs = re.findall(r'\(([^)]*)\)\s*Tj', s)
    for t in tjs:
        t = t.strip()
        if len(t) > 3 and '\\' not in t:
            all_text.append(t)

# Strategy 3: Look for text between BT/ET more broadly
bt_blocks = re.findall(r'BT\n(.*?)\nET', text, re.DOTALL)
print(f'Found {len(bt_blocks)} BT/ET blocks')

# Strategy 4: Look for text in parentheses with Tj
all_tj = re.findall(r'\(([^)]+)\)\s*Tj', text)
print(f'Found {len(all_tj)} Tj matches')

# Strategy 5: Look for hex encoded text
hex_texts = re.findall(r'<([0-9A-Fa-f]+)>\s*Tj', text)
print(f'Found {len(hex_texts)} hex Tj matches')

if all_tj:
    with open(r'd:\papers-code\Extremum_prevention\paper_text.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_tj[:500]))
    print('Wrote Tj text to paper_text.txt')

if hex_texts:
    decoded = []
    for h in hex_texts[:200]:
        try:
            decoded.append(bytes.fromhex(h).decode('utf-16-be', errors='ignore'))
        except:
            pass
    if decoded:
        with open(r'd:\papers-code\Extremum_prevention\paper_hex.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(decoded))
        print(f'Wrote {len(decoded)} hex-decoded texts')