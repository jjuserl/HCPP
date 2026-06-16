import re

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

text = content.decode('latin-1', errors='ignore')

# Find BT/ET text blocks
blocks = re.findall(r'BT(.*?)ET', text, re.DOTALL)
all_text = []
for b in blocks:
    tj = re.findall(r'\((.*?)\)\s*Tj', b)
    for t in tj:
        t = t.strip()
        if len(t) > 3:
            all_text.append(t)

result = '\n'.join(all_text)
with open(r'd:\papers-code\Extremum_prevention\paper_text.txt', 'w', encoding='utf-8') as f:
    f.write(result)

print(f'Extracted {len(all_text)} text segments')