"""
DeepCode - AI Research Engine

Streamlit Web Interface Main Application File
"""

import logging
import os
import sys
import weave

# Disable .pyc file generation
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import UI modules
from ui.layout import main_layout
from utils.experiment_tracker import get_experiment_tracker


def main():
    """
    Main function - Streamlit application entry

    All UI logic has been modularized into ui/ folder
    """
    # Initialize experiment tracker based on configuration
    try:
        tracker = get_experiment_tracker()
    except Exception as e:
        print(f"Error initializing experiment tracker: {e}")
        
        to
    # Run main layout
    sidebar_info = main_layout()

    return sidebar_info


if __name__ == "__main__":
    main()
