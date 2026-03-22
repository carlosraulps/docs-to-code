import os
import re
from typing import List

def clean_latex_for_markdown(content: str) -> str:
    """
    Converts basic LaTeX structural elements to Markdown.
    Leaves math environments ($...$, $$...$$, \\begin{equation}) intact
    as most Markdown renderers support them natively.
    """
    # Convert headings
    content = re.sub(r'\\section\*?\{(.*?)\}', r'# \1', content)
    content = re.sub(r'\\subsection\*?\{(.*?)\}', r'## \1', content)
    content = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'### \1', content)
    
    # Text formatting
    content = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', content)
    content = re.sub(r'\\textit\{(.*?)\}', r'*\1*', content)
    content = re.sub(r'\\emph\{(.*?)\}', r'*\1*', content)
    
    # Itemize/Enumerate simple conversion (basic)
    # This is rudimentary but helps clean up the markdown view
    content = content.replace(r'\begin{itemize}', '')
    content = content.replace(r'\end{itemize}', '')
    content = content.replace(r'\begin{enumerate}', '')
    content = content.replace(r'\end{enumerate}', '')
    content = re.sub(r'\\item\s+', r'- ', content)
    
    return content

from pathlib import Path

def generate_md_file(title: str, content_list: List[str], output_dir: str | Path) -> str:
    """
    Generates a .md file for the given title and content.
    Returns the path to the generated file.
    """
    output_dir = Path(output_dir)
    filename = f"{title}.md"
    output_path = output_dir / filename

    try:
        with open(output_path, 'w') as f:
            f.write(f"<!-- Auto-generated Markdown for {title} -->\n")
            f.write(f"# {title}\n\n")
            
            for i, content in enumerate(content_list):
                f.write(f"<!-- --- Page {i+1} --- -->\n")
                f.write(content)
                f.write("\n\n---\n\n")
        
        return str(output_path)
    except Exception as e:
        print(f"Error writing Markdown file {output_path}: {e}")
        return ""
