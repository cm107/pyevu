import importlib

def require_dependencies(dependencies: list[str]):
    if type(dependencies) is str:
        dependencies = [dependencies]
    for import_name in dependencies:
        if importlib.util.find_spec(import_name) is None:
            raise ImportError(f"Need to install {import_name} in order to run this method.")