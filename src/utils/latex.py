import os
import sys
from typing import List

from pathlib import Path

def generate_tex_file(title: str, content_list: List[str], output_dir: str | Path) -> str:
    """
    Generates a .tex file for the given title and content.
    The content_list contains LaTeX strings (one per image/page).
    Returns the path to the generated file.
    """
    output_dir = Path(output_dir)
    filename = f"{title}.tex"
    output_path = output_dir / filename

    try:
        with open(output_path, 'w') as f:
            f.write(f"% Auto-generated LaTeX for {title}\n")
            f.write(f"\\section*{{{title}}}\n\n")
            
            for i, content in enumerate(content_list):
                f.write(f"% --- Page {i+1} ---\n")
                f.write(content)
                f.write("\n\n")
        
        return output_path
    except Exception as e:
        print(f"Error writing LaTeX file {output_path}: {e}", file=sys.stderr)
        return ""

def get_packages_block() -> str:
    """
    Returns a string containing the standard required packages.
    """
    return """
% Required Packages for Compilation:
% \\usepackage{amsmath}
% \\usepackage{amssymb}
% \\usepackage{amsthm}
% \\usepackage{graphicx}
% \\usepackage{hyperref}
% \\usepackage[utf8]{inputenc}
% \\usepackage[T1]{fontenc}
% \\usepackage{geometry}
% \\usepackage{float}
% Or other possible packages.
"""
