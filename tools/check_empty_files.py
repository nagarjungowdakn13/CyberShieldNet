"""Check repository for empty or effectively-empty Python files.

This script prints Python files that have fewer than `min_lines`
non-comment, non-blank lines. Run from the repository root.
"""
from pathlib import Path
import re

def is_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith('#'):
        return False
    # simple docstring detection (lines that are only triple-quoted strings)
    if re.match(r"^('{3}|\"{3}).*\1$", s):
        return False
    return True

def scan(root: Path, min_lines: int = 3):
    py_files = list(root.rglob('*.py'))
    empties = []
    for p in py_files:
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue

        lines = text.splitlines()
        code_lines = [l for l in lines if is_code_line(l)]
        if len(code_lines) < min_lines:
            empties.append((p, len(code_lines)))

    return empties

if __name__ == '__main__':
    import sys
    root = Path('.')
    empties = scan(root)
    if not empties:
        print('No effectively-empty Python files found.')
        sys.exit(0)

    print('Potentially empty or minimal Python files (path, code_lines):')
    for p, n in sorted(empties):
        print(f'{p} -> {n}')
    sys.exit(0)
