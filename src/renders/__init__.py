import os
import importlib
from pathlib import Path

# Get the directory containing this __init__.py
directory = os.path.dirname(__file__)

# Iterate over files in the directory
for filename in os.listdir(directory):
    # if filename != "__init__.py":
    if "__" not in filename:
        module_name = f"renders.{filename[:-3]}"  # stats.py, main.py
        function_name = f"print_{filename[:-3].split('_')[-1]}"  # e.g., `print_stats`, `print_main`

        # Import the module
        module = importlib.import_module(module_name)

        # Add function to the package namespace if it exists
        if hasattr(module, function_name):
            globals()[function_name] = getattr(module, function_name)


