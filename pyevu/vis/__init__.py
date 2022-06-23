import importlib

for import_name in ['matplotlib', 'numpy']:
    if importlib.util.find_spec(import_name) is None:
        raise ImportError(f"Need to install {import_name} in order to run this script.")