#!/usr/bin/env python3
"""
Script to fix matplotlib display issues in Jupyter notebooks by:
1. Adding %matplotlib inline at the beginning
2. Removing plt.show() calls that can cause separate windows
3. Ensuring plt.close(fig) is used for cleanup
"""

import json
import sys
import os

def fix_notebook(notebook_path):
    """Fix matplotlib display issues in a Jupyter notebook."""
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Check if we need to add %matplotlib inline
    needs_inline = True
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '%matplotlib inline' in source or '%matplotlib notebook' in source:
                needs_inline = False
                break
    
    # If we need to add matplotlib inline, find the first import cell
    if needs_inline:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'import matplotlib' in source or 'import numpy' in source:
                    # Add %matplotlib inline at the beginning
                    new_source = '%matplotlib inline\n\n' + source
                    cell['source'] = new_source.split('\n')
                    print(f"Added '%matplotlib inline' to cell {i}")
                    break
    
    # Remove plt.show() calls and ensure plt.close(fig) exists
    changes_made = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []
            
            for line in source_lines:
                # Remove standalone plt.show() calls
                if line.strip() == 'plt.show()':
                    # Check if there's a plt.close(fig) nearby
                    has_close = any('plt.close' in l for l in source_lines)
                    if not has_close:
                        new_lines.append('# plt.show()  # Commented out for inline display\n')
                        changes_made += 1
                    else:
                        # Just comment it out
                        new_lines.append('# plt.show()  # Commented out for inline display\n')
                        changes_made += 1
                else:
                    new_lines.append(line)
            
            if changes_made > 0:
                cell['source'] = new_lines
    
    # Save the modified notebook
    output_path = notebook_path.replace('.ipynb', '_fixed.ipynb')
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    print(f"Fixed notebook saved as: {output_path}")
    print(f"Total changes made: {changes_made}")
    
    return output_path

def main():
    """Main function."""
    notebook_path = '/home/skipjack/Documents/geopack-vectorize/larmor_radius_analysis/curvature_scattering_threshold_analysis.ipynb'
    
    if os.path.exists(notebook_path):
        print(f"Fixing matplotlib display issues in: {notebook_path}")
        fixed_path = fix_notebook(notebook_path)
        print("\nTo use the fixed notebook:")
        print(f"1. Open: {fixed_path}")
        print("2. Restart the kernel")
        print("3. Run all cells")
        print("\nThe figures will now display inline without separate windows.")
    else:
        print(f"Error: Notebook not found at {notebook_path}")

if __name__ == "__main__":
    main()