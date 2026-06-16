import re
import zlib

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

# Find the content stream objects (8 0 R, 9 0 R, 10 0 R referenced in page 2 0 R)
# Let's find all objects with FlateDecode
# Strategy: Find each object, check if it has /Filter /FlateDecode, then decompress

# First, find all objects
obj_pattern = re.compile(rb'(\d+ \d+ obj.*?endobj)', re.DOTALL)
objects = obj_pattern.findall(content)

print(f'Found {len(objects)} objects')

all_text = []
for i, obj in enumerate(objects):
    obj_str = obj.decode('latin-1', errors='ignore')
    if '/FlateDecode' in obj_str and '/Length' in obj_str:
        # Find the stream content
        stream_match = re.search(rb'stream\r?\n(.*?)endstream', obj, re.DOTALL)
        if stream_match:
            stream_data = stream_match.group(1)
            try:
                decompressed = zlib.decompress(stream_data)
                decoded = decompressed.decode('latin-1', errors='ignore')
                all_text.append((i, decoded))
            except Exception as e:
                all_text.append((i, f'[Decompress error: {e}]'))
        else:
            # Try without the newline after stream
            stream_match = re.search(rb'stream\n(.*?)\nendstream', obj, re.DOTALL)
            if stream_match:
                stream_data = stream_match.group(1)
                try:
                    decompressed = zlib.decompress(stream_data)
                    decoded = decompressed.decode('latin-1', errors='ignore')
                    all_text.append((i, decoded))
                except:
                    pass

with open(r'd:\papers-code\Extremum_prevention\paper_decoded.txt', 'w', encoding='utf-8') as f:
    for idx, text in all_text:
        f.write(f'\n=== Object {idx} (len={len(text)}) ===\n')
        f.write(text[:5000])
        f.write('\n')

print(f'Wrote {len(all_text)} decompressed objects to paper_decoded.txt')