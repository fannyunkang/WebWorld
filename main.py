# main.py
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from core.tasks.base_task import TaskFactory
print("Registry Task:", TaskFactory._tasks.keys())
from core.pipeline import Pipeline
from core.seed import set_global_seed

set_global_seed(42)

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    tasks_config = load_yaml_config("./config/task_config.yaml")
    global_config = load_yaml_config("./config/global_config.yaml")
    model_config = load_yaml_config("./config/model_config.yaml")

    selected_tasks = global_config.get("selected_tasks", [])
    if not selected_tasks:
        print("Warning: No tasks selected, the program will exit.")
        sys.exit(1)

    # Prepare task configurations
    task_configs = []
    for task_info in selected_tasks:
        task_name = task_info["task_path"].split("/")[0]  # Main task name
        task_args = tasks_config.get(task_name, {}).copy()
        task_args.update(task_info["args"])  # Merge general task parameters with specific parameters
        task_args["task_path"] = task_info["task_path"]  # Full task path
        task_args["task_name"] = task_name
        task_configs.append(task_args)

    # Result save path
    model_name = model_config["model"]
    # Construct model-specific result directory path
    output_dir = Path("./results") / model_name
    # Create directory (if it doesn't exist)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Construct full file path (using model name as file name)
    output_file_path = output_dir / (model_name + ".jsonl")
    # The Pipeline class may expect a string path, so convert accordingly
    output_file = str(output_file_path)

    # Create and run the Pipeline
    pipeline = Pipeline(task_configs, model_config, output_file=output_file, global_config=global_config)
    pipeline.run_all()
    print(f"All tasks completed. Results have been saved to {output_file}")
