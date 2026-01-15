import os

file_path = 'app/services/zimage_generator.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_manual_loop = False
indent_block = False
block_end = False

# Constants to help identify where to start indenting
START_MARKER = '        # Unpack pipeline components'
# This ends the indented block
END_MARKER = '        return PILImage.fromarray(image[0])'

header_inserted = False

for i, line in enumerate(lines):
    if 'def _manual_generation_loop' in line:
        in_manual_loop = True
    
    if in_manual_loop and not header_inserted:
        if START_MARKER in line:
            # We found the start of the logic block.
            # Insert the context manager
            new_lines.append('        # CRITICAL: Disable gradient calculation for inference to prevent VRAM leak\n')
            new_lines.append('        with torch.inference_mode():\n')
            header_inserted = True
            indent_block = True
    
    if indent_block:
        # Check if we reached the end
        if END_MARKER in line:
            # Indent this last line too
            new_lines.append('    ' + line)
            indent_block = False
            in_manual_loop = False # Done
        else:
            # Indent the line
            if line.strip() == "":
                new_lines.append(line) # Maintain empty lines
            else:
                new_lines.append('    ' + line)
    else:
        # Just copy the line
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully wrapped _manual_generation_loop in torch.inference_mode()")
