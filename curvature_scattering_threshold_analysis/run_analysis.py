#!/usr/bin/env python3
"""
Script to run the curvature scattering analysis and generate all figures.

Usage:
    python run_analysis.py
"""

import sys
import os
import time

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("CURVATURE SCATTERING THRESHOLD ANALYSIS")
print("Running analysis to generate all figures...")
print("=" * 80)
print()

start_time = time.time()

try:
    # Import and run the main analysis
    from curvature_scattering_analysis import main
    
    # Run the analysis
    main()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print()
    print("=" * 80)
    print(f"Analysis completed successfully in {minutes}m {seconds}s!")
    print("All figures have been saved to the 'figures' directory.")
    print("See README.md for the complete analysis report.")
    print("=" * 80)
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running this from the curvature_scattering_threshold_analysis directory")
    print("and that geopack is properly installed.")
    sys.exit(1)
    
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)