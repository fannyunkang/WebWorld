import json
import os
import shutil
import threading
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.runner import TaskRunner
from core.tasks.base_task import TaskFactory
from core.seed import generate_unique_id
from tqdm import tqdm
from collections import defaultdict
import time
import yaml

class Pipeline:
    def __init__(self, task_configs: List[Dict], model_config: Dict, output_file: str, global_config: Dict = None):
        self.task_configs = task_configs
        self.model_config = model_config
        self.output_file = output_file
        self.global_config = global_config or {}
        # --- Files and Locks ---
        # For Inference
        self.log_file = self.output_file + ".infer.log" # More specific name
        self._infer_log_lock = threading.Lock()
        # For Evaluation
        self.eval_log_file = self.output_file + ".eval.log" # Specific log for evaluation
        self._eval_log_lock = threading.Lock()
        # --- End Files and Locks ---
        self.task_runners = {}
        print("Initializing TaskRunners...")
        for task_config in self.task_configs:
            task_path = task_config['task_path']
            try:
                self.task_runners[task_path] = TaskRunner(task_config, model_config)
            except Exception as e:
                print(f"Error initializing TaskRunner for {task_path}: {e}")
                # Decide how to handle: raise error, skip, etc.
                # For now, we'll allow proceeding but log the error.
        print(f"Initialized {len(self.task_runners)} TaskRunners.")


    def generate_prompts(self) -> List[Dict]:
        """为所有任务生成 prompts 写入文件 (only if main file doesn't exist)."""
        data = []

        # Check if output file exists - avoid overwriting if it does.
        # Inference/Evaluation should handle resuming from existing files.
        if os.path.exists(self.output_file):
             print(f"Main output file '{self.output_file}' already exists.")
             print("Skipping prompt generation. Assuming file contains tasks or previous results.")
             return [] # Prevent overwriting

        # Also check for log files from incomplete previous runs
        if os.path.exists(self.log_file) or os.path.exists(self.eval_log_file):
            print(f"Warning: Log files ('{self.log_file}', '{self.eval_log_file}') exist.")
            print("Skipping prompt generation. Please ensure previous runs are resolved or logs cleared if starting fresh.")
            return []


        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"Generating prompts and writing initial tasks to {self.output_file}...")
        for task_config in tqdm(self.task_configs, desc="Generating Task Path"):
            task_path = task_config['task_path']
            sample_num = task_config.get('sample_num', 1)
            task_runner = self.task_runners.get(task_path)
            if not task_runner:
                print(f"\nWarning: TaskRunner not found for {task_path} during prompt generation. Skipping.")
                continue

            for i in range(sample_num):
                sample_id = generate_unique_id(task_path, i)
                data_item = {
                    "sample_id": sample_id,
                    "prompt": None, # Initialize as None
                    "metadata": None, # Initialize as None
                    "answer": None, # Initialize as None
                    "evaluation_results": None, # Initialize as None
                    "task": f"longeval/{task_path}",
                    "source": "longeval",
                    "eval_args": self.model_config.get('params', {}),
                    "task_config": task_config,
                }
                try:
                    prompt, metadata = task_runner.generate_prompt(**data_item.copy())
                    data_item["prompt"] = prompt
                    data_item["metadata"] = metadata
                except Exception as e:
                     print(f"\nError generating prompt for {sample_id}: {e}")
                     data_item["prompt"] = "ERROR: Prompt generation failed"
                     data_item["metadata"] = {"error": str(e)}
                data.append(data_item)

        # Write initial prompts file
        try:
            # Use the safe rewrite helper for consistency, even for initial write
            self._safe_rewrite_file(data, self.output_file)
            print(f"Successfully generated and saved {len(data)} initial tasks.")
        except Exception as e:
            print(f"Error writing initial prompts to {self.output_file}: {e}")
            return []

        return data

    # --- Inference Section (Append-Only Log) ---

    def _load_and_prepare_inference_tasks(self):
        """Loads state for inference, returning items_map and tasks_to_process."""
        items_map = {}
        processed_ids = set() # IDs with valid 'answer'

        def _parse_infer(line, line_num, filename):
            try:
                item = json.loads(line)
                item_id = item.get("sample_id")
                if not item_id: return # Skip items without ID

                items_map[item_id] = item # Log overrides main
                answer = item.get("answer")
                is_error = isinstance(answer, str) and answer.startswith("ERROR:")
                is_valid_answer = answer is not None and answer != "" and not is_error

                if is_valid_answer:
                    processed_ids.add(item_id)
                elif item_id in processed_ids: # Log entry invalidates previous valid answer
                    processed_ids.discard(item_id)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON in '{filename}' line {line_num}")
            except Exception as e:
                print(f"Error processing line {line_num} in '{filename}': {e}")

        # 1. Load from main file
        if os.path.exists(self.output_file):
            print(f"Reading main file: {self.output_file}")
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip(): _parse_infer(line, i + 1, self.output_file)
            except Exception as e: return None, None # Critical read error
        else:
             print(f"Main file {self.output_file} not found.")

        # 2. Load from inference log file
        if os.path.exists(self.log_file):
            print(f"Reading inference log: {self.log_file}")
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip(): _parse_infer(line, i + 1, self.log_file)
            except Exception as e: return None, None # Critical read error

        if not items_map and not os.path.exists(self.output_file):
            print("Error: No input data found for inference.")
            return None, None

        # 3. Identify tasks
        tasks_to_process = []
        retry_errors = True # Or make configurable
        for item_id, item in items_map.items():
            if item_id not in processed_ids:
                is_error = isinstance(item.get("answer"), str) and item["answer"].startswith("ERROR:")
                if not is_error or (is_error and retry_errors):
                    # Skip if prompt generation failed earlier
                    if item.get("prompt") != "ERROR: Prompt generation failed":
                        tasks_to_process.append(item)
                        if is_error and retry_errors:
                             print(f"Scheduling retry inference for item {item_id}")
                    # else: Keep the prompt error status

        print(f"Loaded {len(items_map)} items for inference.")
        print(f"Found {len(processed_ids)} items with valid answers.")
        print(f"Need to run inference for {len(tasks_to_process)} items.")
        return items_map, tasks_to_process

    def _append_to_infer_log(self, item):
        """Thread-safely appends inference result to the inference log file."""
        item_id = item.get("sample_id")
        if not item_id: return
        try:
            json_string = json.dumps(item, ensure_ascii=False)
            with self._infer_log_lock: # Use the inference lock
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json_string + "\n")
        except Exception as e:
            print(f"\n[Error] Failed to append item {item_id} to inference log {self.log_file}: {e}")

    def _merge_infer_log_to_main(self, final_items_map):
        """Merges the inference log file into the main output file."""
        if not final_items_map:
             print("No items found in memory to merge for inference.")
             return

        print(f"\nMerging inference results into main file: {self.output_file}...")
        try:
            # Use the safe rewrite helper
            self._safe_rewrite_file(list(final_items_map.values()), self.output_file) # Pass list of items
            print(f"Main file updated with inference results: {self.output_file}")

            # Remove inference log file after successful merge
            if os.path.exists(self.log_file):
                try:
                    os.remove(self.log_file)
                    print(f"Inference log file removed: {self.log_file}")
                except OSError as e:
                    print(f"Warning: Could not remove inference log file {self.log_file}: {e}")
        except Exception as e:
             print(f"\n[Error] Failed during inference merge process: {e}")
             print(f"Inference log file ({self.log_file}) was NOT removed.")


    def run_inference_multithread(self, MAX_WORKERS=None):
        # Get inference workers from global config, fallback to default if not specified
        if MAX_WORKERS is None:
            MAX_WORKERS = self.global_config.get('threading_config', {}).get('inference_workers', 8)
        
        print("--- Running Inference ---")
        items_map, tasks_to_process = self._load_and_prepare_inference_tasks()

        if items_map is None: return # Loading failed
        if not tasks_to_process:
            print("No tasks require inference processing.")
            if os.path.exists(self.log_file): self._merge_infer_log_to_main(items_map)
            print("Inference step finished (no new tasks).")
            return

        total_tasks_to_run = len(tasks_to_process)
        print(f"Starting inference for {total_tasks_to_run} tasks with {MAX_WORKERS} workers...")

        def _infer_worker(item_data):
            item_copy = item_data.copy()
            item_id = item_copy.get("sample_id", "UNKNOWN_ID")
            start_time = time.time()
            try:
                task_path = item_copy.get('task_config', {}).get('task_path')
                if not task_path: raise ValueError("Missing 'task_config.task_path'")
                task_runner = self.task_runners.get(task_path)
                if not task_runner: raise ValueError(f"TaskRunner not found for path '{task_path}'")
                prompt = item_copy.get('prompt')
                if prompt is None: raise ValueError("Missing 'prompt'")
                if prompt == "ERROR: Prompt generation failed": # Check added here too
                     item_copy["answer"] = "ERROR: Skipped due to prompt generation failure"
                     return item_copy
                eval_args = item_copy.get('eval_args', {})
                response, _ = task_runner.call_api(prompt, eval_args)
                item_copy["answer"] = response
            except Exception as e:
                error_msg = f"ERROR: API call failed for item {item_id} - {type(e).__name__}: {str(e)}"
                item_copy["answer"] = error_msg
                print(f"\n[Worker Error] Item {item_id}: {error_msg}")
            finally: # <--- NEW: Use finally to ensure duration is always recorded
                item_copy["inference_duration_sec"] = time.time() - start_time
                # <--- END NEW
            return item_copy

        completed_count = 0
        run_successful = True
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="InferWorker") as executor:
            futures = {executor.submit(_infer_worker, task.copy()): task.get("sample_id") for task in tasks_to_process}
            progress = tqdm(total=total_tasks_to_run, desc="Running Inference", unit="item", dynamic_ncols=True)
            try:
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        processed_item = future.result()
                        self._append_to_infer_log(processed_item) # Log to inference log
                        if item_id: items_map[item_id] = processed_item # Update map
                        completed_count += 1
                        progress.update(1)
                    except Exception as exc:
                        run_successful = False
                        print(f"\n[Error] Handling result for item ID '{item_id}': {exc}")
                        if item_id and item_id in items_map:
                             items_map[item_id]['answer'] = f"ERROR: Result handling failed - {exc}"
                             self._append_to_infer_log(items_map[item_id])
                        progress.update(1)
            except KeyboardInterrupt: run_successful = False; print("\n[Interrupted] Inference interrupted.")
            except Exception as e: run_successful = False; print(f"\n[Error] Pool execution error: {e}")
            finally: progress.close()

        print(f"\nProcessed {completed_count} items during inference.")
        if run_successful:
            print("Inference run completed. Merging results...")
            self._merge_infer_log_to_main(items_map)
        else:
            print("Inference run finished with errors/interruption.")
            print(f"Completed results logged to: {self.log_file}")
            print(f"Main file ({self.output_file}) may be incomplete.")
            print("Re-run or manually merge if needed.")
        print("--- Inference Step Finished ---")


    # --- Evaluation Section (Append-Only Log) ---

    def _load_for_evaluation(self):
        """Loads state for evaluation, returning items map and tasks needing evaluation."""
        items_map = {}
        evaluated_ids = set() # IDs with valid 'evaluation_results'

        def _parse_eval(line, line_num, filename):
            try:
                item = json.loads(line)
                item_id = item.get("sample_id")
                if not item_id: return

                # Update map (eval log overrides main file's eval results)
                if item_id in items_map:
                    # Preserve existing fields, only update eval results if present in log item
                    if "evaluation_results" in item:
                        items_map[item_id]["evaluation_results"] = item["evaluation_results"]
                else:
                    items_map[item_id] = item

                # Check if evaluation is complete and valid
                eval_results = items_map[item_id].get("evaluation_results")
                is_valid_eval = isinstance(eval_results, dict) and not eval_results.get("error")

                if is_valid_eval:
                    evaluated_ids.add(item_id)
                elif item_id in evaluated_ids: # Log entry invalidates previous eval
                    evaluated_ids.discard(item_id)

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON in '{filename}' line {line_num}")
            except Exception as e:
                print(f"Error processing line {line_num} in '{filename}': {e}")

        # 1. Load from main file (MUST exist and contain inference results)
        if not os.path.exists(self.output_file):
            print(f"Error: Main file '{self.output_file}' not found for evaluation.")
            return None, None
        print(f"Reading main file for evaluation state: {self.output_file}")
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip(): _parse_eval(line, i + 1, self.output_file)
        except Exception as e:
             print(f"Error reading main file for evaluation: {e}")
             return None, None # Critical read error

        # 2. Load from evaluation log file
        if os.path.exists(self.eval_log_file):
            print(f"Reading evaluation log: {self.eval_log_file}")
            try:
                with open(self.eval_log_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip(): _parse_eval(line, i + 1, self.eval_log_file)
            except Exception as e:
                 print(f"Error reading evaluation log file: {e}")
                 return None, None # Critical read error

        # 3. Identify tasks needing evaluation
        tasks_to_evaluate = []
        retry_eval_errors = False # Set True to retry evaluations marked with {"error": ...}

        for item_id, item in items_map.items():
            # Condition 1: Does it *have* a valid answer to evaluate?
            answer = item.get("answer")
            has_valid_answer = answer is not None and answer != "" and not (isinstance(answer, str) and answer.startswith("ERROR:"))

            if has_valid_answer:
                # Condition 2: Is it already evaluated?
                 if item_id not in evaluated_ids:
                     eval_results = item.get("evaluation_results")
                     is_eval_error = isinstance(eval_results, dict) and eval_results.get("error")
                     # Add to list if not evaluated OR if it's an error and we retry errors
                     if not is_eval_error or (is_eval_error and retry_eval_errors):
                          tasks_to_evaluate.append(item)
                          if is_eval_error and retry_eval_errors:
                               print(f"Scheduling retry evaluation for item {item_id}")

        print(f"Loaded {len(items_map)} items for evaluation.")
        print(f"Found {len(evaluated_ids)} items already evaluated.")
        print(f"Need to run evaluation for {len(tasks_to_evaluate)} items.")
        return items_map, tasks_to_evaluate


    def _append_to_eval_log(self, item):
        """Safely appends evaluation result to the evaluation log file."""
        item_id = item.get("sample_id")
        if not item_id: return
        try:
            json_string = json.dumps(item, ensure_ascii=False)
            with self._eval_log_lock: # Use the evaluation lock
                with open(self.eval_log_file, 'a', encoding='utf-8') as f:
                    f.write(json_string + "\n")
        except Exception as e:
            print(f"\n[Error] Failed to append item {item_id} to evaluation log {self.eval_log_file}: {e}")


    def _merge_eval_log_to_main(self, final_items_map):
        """Merges the evaluation log file into the main output file."""
        if not final_items_map:
             print("No items found in memory to merge for evaluation.")
             return

        print(f"\nMerging evaluation results into main file: {self.output_file}...")
        try:
            # Use the safe rewrite helper
            self._safe_rewrite_file(list(final_items_map.values()), self.output_file)
            print(f"Main file updated with evaluation results: {self.output_file}")

            # Remove evaluation log file after successful merge
            if os.path.exists(self.eval_log_file):
                try:
                    os.remove(self.eval_log_file)
                    print(f"Evaluation log file removed: {self.eval_log_file}")
                except OSError as e:
                    print(f"Warning: Could not remove evaluation log file {self.eval_log_file}: {e}")
        except Exception as e:
             print(f"\n[Error] Failed during evaluation merge process: {e}")
             print(f"Evaluation log file ({self.eval_log_file}) was NOT removed.")

    def evaluate_and_save(self, MAX_WORKERS=None):
        # Get evaluation workers from global config, fallback to default if not specified
        if MAX_WORKERS is None:
            MAX_WORKERS = self.global_config.get('threading_config', {}).get('evaluation_workers', 4)
        
        print("--- Running Evaluation (Multi-threaded) ---")
        # Pre-check: Ensure inference log is merged or absent
        if os.path.exists(self.log_file):
             print("Warning: Inference log file exists. Attempting merge before evaluation.")
             infer_map, _ = self._load_and_prepare_inference_tasks()
             if infer_map is None:
                  print("Error loading data for pre-evaluation inference merge. Aborting evaluation.")
                  return
             self._merge_infer_log_to_main(infer_map)
             if os.path.exists(self.log_file): # Check if merge failed
                  print("Error: Failed to merge inference log before evaluation. Aborting.")
                  return
             print("Inference log merged successfully.")

        # 1. Load state for evaluation
        items_map, tasks_to_evaluate = self._load_for_evaluation()

        if items_map is None: return # Loading failed
        if not tasks_to_evaluate:
            print("No tasks require evaluation.")
            if os.path.exists(self.eval_log_file): self._merge_eval_log_to_main(items_map)
            print("Evaluation step finished (no new tasks).")
            return

        total_tasks_to_run = len(tasks_to_evaluate)
        print(f"Starting evaluation for {total_tasks_to_run} tasks with {MAX_WORKERS} workers...")

        # 2. Define the worker function for a single evaluation task
        def _eval_worker(item_data):
            item_copy = item_data.copy()
            item_id = item_copy.get("sample_id", "UNKNOWN_ID")
            start_time = time.time()
            try:
                task_path = item_copy.get('task_config', {}).get('task_path')
                if not task_path: raise ValueError("Missing 'task_config.task_path'")
                task_runner = self.task_runners.get(task_path)
                if not task_runner: raise ValueError(f"TaskRunner not found for path '{task_path}'")
                
                answer = item_copy.get("answer")
                eval_result = task_runner.evaluate_response(answer, item_copy)
                item_copy["evaluation_results"] = eval_result
            except Exception as e:
                error_msg = f"ERROR: Evaluation failed for item {item_id} - {type(e).__name__}: {str(e)}"
                item_copy["evaluation_results"] = {"error": error_msg}
                print(f"\n[Eval Worker Error] Item {item_id}: {error_msg}")
            finally:
                item_copy["evaluation_duration_sec"] = time.time() - start_time
            return item_copy

        # 3. Process tasks in parallel using ThreadPoolExecutor
        completed_count = 0
        run_successful = True
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="EvalWorker") as executor:
            futures = {executor.submit(_eval_worker, task.copy()): task.get("sample_id") for task in tasks_to_evaluate}
            progress = tqdm(total=total_tasks_to_run, desc="Running Evaluation", unit="item", dynamic_ncols=True)
            try:
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        processed_item = future.result()
                        self._append_to_eval_log(processed_item)
                        if item_id: items_map[item_id] = processed_item
                        completed_count += 1
                        progress.update(1)
                    except Exception as exc:
                        run_successful = False
                        print(f"\n[Error] Handling evaluation result for item ID '{item_id}': {exc}")
                        if item_id and item_id in items_map:
                             items_map[item_id]['evaluation_results'] = {"error": f"Result handling failed - {exc}"}
                             self._append_to_eval_log(items_map[item_id])
                        progress.update(1)
            except KeyboardInterrupt: run_successful = False; print("\n[Interrupted] Evaluation interrupted.")
            except Exception as e: run_successful = False; print(f"\n[Error] Pool execution error: {e}")
            finally: progress.close()

        # 4. Final Merge
        print(f"\nProcessed {completed_count} items during evaluation.")
        if run_successful:
            print("Evaluation run completed. Merging results...")
            self._merge_eval_log_to_main(items_map)
        else:
            print("Evaluation run finished with errors/interruption.")
            print(f"Completed evaluation results logged to: {self.eval_log_file}")
            print(f"Main file ({self.output_file}) may not reflect latest evaluations.")
            print("Re-run or manually merge if needed.")
        print("--- Evaluation Step Finished ---")


    # --- Analysis Section (Reads final merged file) ---
    def analyze_hierarchical_metrics(self):
        print("--- Running Analysis ---")
        if os.path.exists(self.log_file) or os.path.exists(self.eval_log_file):
            print("Error: Log files exist. Analysis requires a clean, merged output file.")
            print("Please ensure previous steps completed successfully or manually merge logs.")
            return

        if not os.path.exists(self.output_file):
            print(f"Error: Output file {self.output_file} not found for analysis.")
            return

        from collections import defaultdict
        from core.tasks.base_task import TaskFactory # Ensure this import is valid

        print(f"Analyzing metrics from: {self.output_file}")
        stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        items_processed = 0
        items_analyzed = 0

        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    items_processed += 1
                    try:
                        item = json.loads(line)
                        inference_duration = item.get("inference_duration_sec")
                        evaluation_duration = item.get("evaluation_duration_sec")
                        eval_results = item.get('evaluation_results')
                        if not isinstance(eval_results, dict) or eval_results.get("error"):
                            continue # Skip items without valid evaluation

                        task_full_path = item.get('task', '')
                        if not task_full_path.startswith('longeval/'): continue
                        task_path_parts = task_full_path.split('/')[1:]
                        if not task_path_parts: continue

                        task_name = task_path_parts[0]
                        task_class = TaskFactory._tasks.get(task_name)
                        if not task_class or not hasattr(task_class, 'get_registered_metrics'): continue
                        metrics_to_collect = task_class.get_registered_metrics()
                        if not metrics_to_collect: continue

                        items_analyzed += 1 # Count items contributing to stats
                        for depth in range(1, len(task_path_parts) + 1):
                            hierarchy = '/'.join(task_path_parts[:depth])
                            for metric in metrics_to_collect:
                                value = eval_results.get(metric)
                                if isinstance(value, (int, float)):
                                    stats[hierarchy][metric][0] += value
                                    stats[hierarchy][metric][1] += 1
                            if isinstance(inference_duration, (int, float)):
                                stats[hierarchy]['inference_duration_sec'][0] += inference_duration
                                stats[hierarchy]['inference_duration_sec'][1] += 1
                            
                            if isinstance(evaluation_duration, (int, float)):
                                stats[hierarchy]['evaluation_duration_sec'][0] += evaluation_duration
                                stats[hierarchy]['evaluation_duration_sec'][1] += 1
                    except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line during analysis.")
                    except Exception as e: print(f"Error analyzing item: {e} - Line: {line[:100]}...")

            if items_processed == 0: print("No items found in file for analysis."); return
            if items_analyzed == 0: print("No items with valid evaluation results found for analysis."); return

            report = defaultdict(dict)
            for hierarchy, metric_data in sorted(stats.items()):
                for metric, (total, count) in sorted(metric_data.items()):
                    if count > 0:
                        report[hierarchy][metric] = {'average': round(total / count, 4), 'samples': count}

            model_name = self.model_config.get("model", "unknown_model")
            base_report_filename = model_name + "_metric_report.json"

            output_dir = os.path.dirname(self.output_file) 

            report_filepath = os.path.join(output_dir, base_report_filename)

            print(f"Saving analysis report ({items_analyzed} items analyzed) to: {report_filepath}") 
            self._safe_rewrite_file(report, report_filepath, is_json_report=True) 
            print("Analysis report saved.")


        except Exception as e:
            print(f"An unexpected error occurred during metric analysis: {e}")
        print("--- Analysis Step Finished ---")


    # --- Utility ---
    def _safe_rewrite_file(self, data, target_file, is_json_report=False):
        """将数据安全地写入目标文件 (JSON Lines or single JSON object)"""
        temp_file = target_file + ".tmp_rewrite"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                if is_json_report:
                    # Write as a single JSON object with indentation
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    # Write as JSON Lines
                    if isinstance(data, list): # Check if data is a list of items
                        for item in data:
                            if isinstance(item, dict): # Ensure item is a dictionary
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            else:
                                print(f"Warning: Skipping non-dict item during rewrite: {type(item)}")
                    else:
                         print(f"Error: Data for JSON Lines rewrite is not a list: {type(data)}")
                         raise TypeError("Data must be a list for JSON Lines output")

            shutil.move(temp_file, target_file)
        except Exception as e:
            print(f"Error safely rewriting file {target_file}: {e}")
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except OSError: pass
            # Re-raise the exception if it's critical for flow control
            raise


    # --- Main Execution ---
    def run_all(self):
        """运行整个流程：Generate -> Infer -> Evaluate -> Analyze"""
        print("--- Starting Pipeline Run ---")
        try:
            self.generate_prompts()
            # Check if we can proceed (initial file exists or was created)
            if not os.path.exists(self.output_file) and not os.path.exists(self.log_file):
                 print("Stopping run: Cannot proceed without initial tasks file.")
                 return

            self.run_inference_multithread() # Use workers from global config
            self.evaluate_and_save() # Use workers from global config
            self.analyze_hierarchical_metrics()
            print("\n--- Pipeline Run Finished Successfully ---")
        except Exception as e:
             print(f"\n--- Pipeline Run Failed ---")
             print(f"Error: {e}")
             import traceback
             traceback.print_exc()
             print("Please check logs and intermediate files.")