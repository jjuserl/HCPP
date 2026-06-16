import re
import zlib

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

# Find all objects
obj_pattern = re.compile(rb'(\d+ \d+ obj.*?endobj)', re.DOTALL)
objects = obj_pattern.findall(content)

# Extract text from all TJ operations
all_text = []
for obj in objects:
    obj_str = obj.decode('latin-1', errors='ignore')
    
    # First decompress if FlateDecode
    if '/FlateDecode' in obj_str:
        stream_match = re.search(rb'stream\r?\n(.*?)(?:\r?\n)?endstream', obj, re.DOTALL)
        if stream_match:
            try:
                obj_str = zlib.decompress(stream_match.group(1)).decode('latin-1', errors='ignore')
            except:
                try:
                    # Try removing trailing bytes
                    data = stream_match.group(1)
                    # Sometimes there's trailing \n before endstream
                    if data.endswith(b'\n'):
                        data = data[:-1]
                    obj_str = zlib.decompress(data).decode('latin-1', errors='ignore')
                except:
                    continue
    
    # Extract text from TJ arrays
    # Pattern: [(text)-number(text)] TJ  or (text) Tj
    tj_matches = re.findall(r'\[(.*?)\]\s*TJ', obj_str)
    for tj in tj_matches:
        # Extract text from the array
        texts = re.findall(r'\(([^)]*)\)', tj)
        line = ''.join(texts)
        if len(line) > 3:
            all_text.append(line)
    
    # Also simple Tj operations
    tj_simple = re.findall(r'\(([^)]+)\)\s*Tj', obj_str)
    for t in tj_simple:
        if len(t) > 3:
            all_text.append(t)

with open(r'd:\papers-code\Extremum_prevention\paper_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_text))

print(f'Extracted {len(all_text)} text lines')