import os

def clean_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    in_frontmatter = False
    modified = False

    while i < len(lines):
        line = lines[i]
        
        if line.strip() == '---':
            if i == 0:
                in_frontmatter = True
            elif in_frontmatter:
                in_frontmatter = False
                
        # Handle frontmatter skipping zero review items
        if in_frontmatter and line.strip().startswith('- asin:'):
            j = i + 1
            block_has_zero = False
            while j < len(lines):
                if lines[j].strip().startswith('- asin:') or lines[j].strip() == '---' or not lines[j].startswith('  '):
                    if lines[j].strip() == '---':
                        pass
                    elif lines[j].strip().startswith('- asin:'):
                        pass
                    else:
                        pass
                
                if lines[j].strip() == 'review_count: 0':
                    block_has_zero = True
                
                if j > i and (lines[j].strip().startswith('- asin:') or lines[j].strip() == '---'):
                    break
                j += 1
                
            if block_has_zero:
                modified = True
                i = j
                continue
                
        # Handle "Pending Spec Review" table rows
        if line.startswith('| **Listing B0') and 'Pending Spec Review' in line:
            modified = True
            i += 1
            continue
            
        # Add separator to comparison table if missing
        if line.startswith('| Product | Best For |'):
            new_lines.append(line)
            # check if the next line is already the separator
            if i + 1 < len(lines) and lines[i+1].startswith('| :---'):
                pass
            else:
                new_lines.append('| :--- | :--- | :--- | :--- | :--- |\n')
                modified = True
            i += 1
            continue
            
        # Skip "Outdoor Utility Light" review sections
        if line.startswith('### Outdoor Utility Light (ASIN:'):
            modified = True
            while i < len(lines):
                if lines[i].startswith('[Check Price on Amazon]'):
                    i += 1
                    if i < len(lines) and lines[i].strip() == '':
                        i += 1
                    break
                i += 1
            continue
            
        new_lines.append(line)
        i += 1

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Cleaned {file_path}")

def main():
    content_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'content')
    for root, _, files in os.walk(content_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    clean_markdown_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    main()
