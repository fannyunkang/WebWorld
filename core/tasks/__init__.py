import importlib
from pathlib import Path

for task_file in Path(__file__).parent.glob("*.py"):
    if task_file.name != "__init__.py" and task_file.name != "base_task.py":
        module_name = task_file.stem
        importlib.import_module(f".{module_name}", package=__name__)