#!/usr/bin/env python3
"""Execute notebook cells to test for errors."""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os

# Change to the notebook directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the notebook
with open('curvature_scattering_threshold_analysis.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Create an execution preprocessor
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

# Execute the notebook
try:
    print("Executing notebook...")
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Notebook executed successfully!")
    
    # Save the executed notebook
    with open('curvature_scattering_threshold_analysis_executed.ipynb', 'w') as f:
        nbformat.write(nb, f)
    print("Saved executed notebook")
    
except Exception as e:
    print(f"Error executing notebook: {e}")
    # Find which cell failed
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
            for output in cell.outputs:
                if output.output_type == 'error':
                    print(f"\nError in cell {i}:")
                    print(output.traceback[-1])
                    break