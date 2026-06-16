import re
import zlib

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

# Find all stream objects with their filters
# Pattern: /Filter ... >> stream ... endstream
text = content.decode('latin-1', errors='ignore')

# Find all streams
stream_pattern = re.compile(r'/Filter\s+/FlateDecode.*?>>\s*\nstream\n(.*?)endstream', re.DOTALL)
streams = stream_pattern.findall(text)
print(f'Found {len(streams)} FlateDecode streams')

# Also look for non-compressed streams
stream_pattern2 = re.compile(r'stream\n(.*?)endstream', re.DOTALL)
all_streams = stream_pattern2.findall(text)
print(f'Found {len(all_streams)} total streams')

all_text = []
for s in streams:
    try:
        # Try to decompress
        data = s.encode('latin-1', errors='ignore')
        # Need to find the actual binary data
        # The stream content in the regex is the text between "stream\n" and "endstream"
        # But it might include trailing \n
        decompressed = zlib.decompress(data)
        decoded = decompressed.decode('latin-1', errors='ignore')
        all_text.append(decoded)
    except Exception as e:
        pass

with open(r'd:\papers-code\Extremum_prevention\paper_decoded.txt', 'w', encoding='utf-8') as f:
    for i, t in enumerate(all_text):
        f.write(f'\n=== Stream {i} ===\n')
        f.write(t)

print(f'Decompressed {len(all_text)} streams, wrote to paper_decoded.txt')