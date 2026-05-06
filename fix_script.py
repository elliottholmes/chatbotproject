with open('train_llm_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find the line after the last import
import_end = None
for i, line in enumerate(lines):
    if line.strip() == 'from datasets import Dataset':
        import_end = i
        break

if import_end is not None:
    # Insert the if __name__ == '__main__': after the import
    lines.insert(import_end + 1, '\nif __name__ == "__main__":\n')
    # Indent all lines after that
    for i in range(import_end + 2, len(lines)):
        lines[i] = '    ' + lines[i]

    with open('train_llm_enhanced.py', 'w') as f:
        f.writelines(lines)
    print('File updated successfully')
else:
    print('Could not find the import line')