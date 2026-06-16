import re
import zlib
import struct

with open(r'd:\papers-code\Extremum_prevention\Hierarchy_Coverage_Path_Planning_With_Proactive_Extremum_Prevention_in_Unknown_Environments.pdf', 'rb') as f:
    content = f.read()

# Find all objects with their streams
# More robust approach: find objects using offsets
text = content.decode('latin-1', errors='ignore')

# Find all stream objects
# Pattern: find "stream\n" and "endstream" pairs
idx = 0
all_text = []
while True:
    start = text.find('stream\n', idx)
    if start == -1:
        break
    # Actual stream content starts after "stream\n"
    start += 7
    end = text.find('endstream', start)
    if end == -1:
        break
    stream_data = content[start:end]  # Use raw bytes
    
    # Check if it's ASCII text (likely not compressed)
    try:
        # Remove trailing \n if present
        if stream_data.endswith(b'\n'):
            stream_data = stream_data[:-1]
        if len(stream_data) < 1000:  # Skip very small streams
            idx = end + 9
            continue
        # Try to decode as text
        decoded = stream_data.decode('latin-1', errors='ignore')
        if len(decoded) == len(stream_data):  # No loss
            all_text.append(decoded)
    except:
        pass
    
    idx = end + 9

# Also try to decompress binary streams
with open(r'd:\papers-code\Extremum_prevention\paper_decoded.txt', 'w', encoding='utf-8') as f:
    for i, t in enumerate(all_text):
        f.write(f'\n=== Stream {i} (len={len(t)}) ===\n')
        f.write(t[:5000])
        f.write('\n')

print(f'Wrote {len(all_text)} streams to paper_decoded.txt')