import re

with open(r'd:\papers-code\Extremum_prevention\paper_decoded.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Extract text from TJ arrays
all_text = []
tj_matches = re.findall(r'\[(.*?)\]\s*TJ', content)
for tj in tj_matches:
    texts = re.findall(r'\(([^)]*)\)', tj)
    line = ''.join(texts)
    if len(line) > 3:
        all_text.append(line)

# Also simple Tj operations
tj_simple = re.findall(r'\(([^)]+)\)\s*Tj', content)
for t in tj_simple:
    if len(t) > 3 and t not in all_text:
        all_text.append(t)

with open(r'd:\papers-code\Extremum_prevention\paper_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_text))

print(f'Extracted {len(all_text)} text lines')