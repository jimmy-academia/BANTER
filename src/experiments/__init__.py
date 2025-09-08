import os
import importlib

# Get the directory containing this __init__.py
directory = os.path.dirname(__file__)

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith("_exp.py") and filename != "__init__.py":
        module_name = f"experiments.{filename[:-3]}"  # `experiments.main_exp`
        function_name = f"run_{filename.split('_')[0]}_exp"  # e.g., `run_main_exp`

        # Import the module
        module = importlib.import_module(module_name)

        # Add function to the package namespace if it exists
        if hasattr(module, function_name):
            globals()[function_name] = getattr(module, function_name)