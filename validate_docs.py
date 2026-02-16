#!/usr/bin/env python3
"""Validate all documentation links."""

import re
import os
from pathlib import Path

def find_markdown_links(content):
    """Extract all markdown links from content."""
    # Match [text](path) pattern
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    return re.findall(pattern, content)

def validate_file(filepath):
    """Validate links in a markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    links = find_markdown_links(content)
    broken_links = []
    
    for text, link in links:
        # Skip external URLs
        if link.startswith('http://') or link.startswith('https://'):
            continue
        
        # Skip anchors
        if link.startswith('#'):
            continue
        
        # Resolve path relative to file location
        file_dir = os.path.dirname(filepath)
        full_path = os.path.normpath(os.path.join(file_dir, link))
        
        # Check if file exists
        if not os.path.exists(full_path):
            broken_links.append((text, link, full_path))
    
    return broken_links

def main():
    """Main validation function."""
    print("Validating documentation links...\n")
    
    # Find all markdown files
    md_files = list(Path('.').rglob('*.md'))
    
    # Exclude archive and .venv directories
    md_files = [f for f in md_files if 'archive' not in str(f) and '.venv' not in str(f)]
    
    total_files = len(md_files)
    total_broken = 0
    
    for md_file in md_files:
        broken = validate_file(md_file)
        if broken:
            print(f"❌ {md_file}:")
            for text, link, full_path in broken:
                print(f"   Broken: [{text}]({link})")
                print(f"   Expected: {full_path}")
            print()
            total_broken += len(broken)
        else:
            print(f"✅ {md_file}")
    
    print(f"\n{'='*60}")
    print(f"Total files checked: {total_files}")
    print(f"Total broken links: {total_broken}")
    
    if total_broken == 0:
        print("✅ All documentation links are valid!")
        return 0
    else:
        print("❌ Some documentation links are broken.")
        return 1

if __name__ == '__main__':
    exit(main())
