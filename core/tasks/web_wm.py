import json
import re
import os
import random
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import jsonlines
from json_repair import repair_json

from core.tasks.base_task import BaseTask, TaskFactory
from core.serve.unified_api import unified_call


class WebWorldModelTask(BaseTask):
    """
    Task for evaluating a web world model on a sample-by-sample basis.
    This version includes robust path handling to avoid FileNotFoundError.
    """
    registered_metrics = [
        'action_effect_accuracy_score', 
        'plausibility_score',
        'total_score'
    ]


    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_path = config.get('task_path', {})
        self.evaluation_model_config = config.get('evaluation_model')
        
        # --- ROBUST PATH HANDLING ---
        # Determine project root based on the current file's location.
        # Assumes this file is at `your_project_root/core/tasks/web_wm_task.py`
        try:
            self.project_root = Path(__file__).resolve().parent.parent.parent
            print(f"[DEBUG] Project root identified as: {self.project_root}")
        except NameError:
            # Fallback for environments where __file__ might not be defined (e.g., some notebooks)
            self.project_root = Path.cwd()
            print(f"[WARN] `__file__` not defined. Using current working directory as project root: {self.project_root}")
        
        self.data_paths = {
            "base": config.get("base"),
            "fine_grained": config.get("fine_grained"),
            "multi_tab": config.get("multi_tab"),
            "long": config.get("long"),
            "xml": config.get("xml"),
            "html": config.get("html"),
            "markdown": config.get("markdown"),
            "playwright": config.get("playwright"),
            "Web2NAL": config.get("Web2NAL"),

            "api_service": config.get("api_service"),
            "code_dev": config.get("code_dev"),
            "game_simulation": config.get("game_simulation"),
            "gui_desktop": config.get("gui_desktop"),
            "web_browser": config.get("web_browser")
        }

        try:
            task_config_path = self.project_root / "config" / "task_config.yaml"
            with open(task_config_path, 'r') as f:
                task_configs = yaml.safe_load(f)

            web_wm_config = task_configs.get("WEB_WORLD_MODEL_EVALUATION", {})

            for key in ["base", "fine_grained", "multi_tab", "long", "xml", "html",
                       "markdown", "playwright", "Web2NAL", "api_service", "code_dev",
                       "game_simulation", "gui_desktop", "web_browser"]:
                if not self.data_paths.get(key):
                    self.data_paths[key] = web_wm_config.get(key)

            print(f"[DEBUG] Loaded data paths from task_config.yaml: {self.data_paths}")
        except Exception as e:
            print(f"[ERROR] Failed to load data paths from task_config.yaml: {e}")
        # ---------------------------
        
        self.samples_map: Dict[str, Dict[str, Any]] = self._load_all_samples_into_map()

        if not self.evaluation_model_config:
            print("[WARN] Evaluation model configuration missing. Using dummy model.")
            self.evaluation_model_config = {'backend': 'dummy', 'model': 'dummy', 'params': {}}

    def _get_task_suffix(self) -> str:
        parts = self.task_path.split('/')
        return parts[-1] if len(parts) > 1 else "base"

    def _is_web2nal_task(self) -> bool:
        """Check if the current task is a Web2NAL task (natural language description)."""
        suffix = self._get_task_suffix()
        return suffix == "Web2NAL"

    def _get_data_path_for_task(self) -> Path:
        """
        Constructs a robust, absolute path to the data file to prevent relative path issues.
        """
        suffix = self._get_task_suffix()
        relative_path_str = self.data_paths.get(suffix)
        
        if not relative_path_str:
            raise ValueError(f"Data path for suffix '{suffix}' is not configured in task_config.yaml.")
        
        # Join the project root with the relative path from the config file.
        absolute_path = self.project_root.joinpath(relative_path_str)
        
        print(f"[DEBUG] Attempting to access data at resolved absolute path: {absolute_path}")
        
        if not absolute_path.is_file():
            error_msg = (
                f"Data file for suffix '{suffix}' not found at resolved absolute path: {absolute_path}\n"
                f"  - Project root was: {self.project_root}\n"
                f"  - Relative path from config was: {relative_path_str}\n"
                f"Please ensure the path in your task_config.yaml is correct relative to the project root and the file exists."
            )
            raise FileNotFoundError(error_msg)
            
        return absolute_path

    def _load_all_samples_into_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all samples from the data file using the resolved absolute path.
        """
        data_path = self._get_data_path_for_task()
        samples_map = {}
        task_prefix = self.task_path
        try:
            # Check file extension to determine how to load the data
            if data_path.suffix == '.json':
                # Handle long.json format - array of trajectories
                with open(data_path, 'r') as f:
                    trajectories = json.load(f)
                    
                sample_num_limit = self.config.get('sample_num')
                if sample_num_limit and isinstance(sample_num_limit, int) and 0 < sample_num_limit < len(trajectories):
                    print(f"[INFO] Limiting loaded samples to the first {sample_num_limit} as per config.")
                    trajectories = trajectories[:sample_num_limit]
                
                # Process each trajectory
                for i, trajectory_data in enumerate(trajectories):
                    # Each trajectory has a "trajectory" array of steps
                    trajectory = trajectory_data.get("trajectory", [])
                    # Store the entire trajectory for multi-turn evaluation
                    sample_id = f"{task_prefix}_{i}"
                    samples_map[sample_id] = {
                        "trajectory": trajectory,
                        "url": trajectory_data.get("url", ""),
                        "site": trajectory_data.get("site", {})
                    }
            else:
                # Handle .jsonl format - one sample per line
                with jsonlines.open(data_path, 'r') as reader:
                    samples_to_load = list(reader)
                    
                    # Limit the number of samples if `sample_num` is specified in the config
                    sample_num_limit = self.config.get('sample_num')
                    if sample_num_limit and isinstance(sample_num_limit, int) and 0 < sample_num_limit < len(samples_to_load):
                        print(f"[INFO] Limiting loaded samples to the first {sample_num_limit} as per config.")
                        samples_to_load = samples_to_load[:sample_num_limit]

                    for i, line in enumerate(samples_to_load):
                        sample_id = f"{task_prefix}_{i}"
                        samples_map[sample_id] = line
            
            print(f"[INFO] Loaded {len(samples_map)} samples for task '{self.task_path}'.")
            return samples_map
        except Exception as e:
            print(f"[ERROR] Failed to load samples from {data_path}: {e}")
            return {}

    def generate_prompt(self, **kwargs) -> Tuple[str, Dict]:
        """
        Generate the prompt text for the task.
        
        Args:
            kwargs: Optional parameters (e.g., sample_id)
            
        Returns:
            Tuple[str, Dict]: The generated prompt text and metadata
        """
        sample_id = kwargs.get('sample_id')
        if not sample_id:
            raise ValueError("generate_prompt requires 'sample_id'.")

        sample = self.samples_map.get(sample_id)
        if not sample:
            print(f"[WARN] Could not find pre-loaded sample data for sample_id '{sample_id}'. Returning empty prompt.")
            return "No sample data found.", {}

        # Check if this is a multi-turn trajectory (from long.json)
        if "trajectory" in sample:
            trajectory = sample["trajectory"]

            is_alternating_format = False
            if len(trajectory) > 0:
                first_item = trajectory[0]
                if ('observation' in first_item and 'action' not in first_item) or \
                   ('action' in first_item and 'observation' not in first_item):
                    is_alternating_format = True

            if is_alternating_format:

                if len(trajectory) < 3:
                    print(f"[WARN] Trajectory too short for sample_id '{sample_id}'. Need at least 3 items.")
                    return "No valid trajectory data.", {}

                init_instruction = (
                    f"You are a world model. I will provide you with an initial state and a sequence of actions. "
                    f"For each action, predict the resulting state.\n"
                    f"Strictly maintain the original format. Output only the full state without explanations, code, or truncation.\n\n"
                )

                trajectory_parts = []
                trajectory_parts.append(f"Initial State:\n{trajectory[0].get('observation', '')}")

                for i in range(1, len(trajectory) - 1, 2):
                    if i < len(trajectory):
                        action = trajectory[i].get('action', '')
                        trajectory_parts.append(f"\nAction: {action}")

                        if i + 1 < len(trajectory) - 1:
                            next_obs = trajectory[i + 1].get('observation', '')
                            trajectory_parts.append(f"\nNext State:\n{next_obs}")

                last_action_idx = len(trajectory) - 2
                if last_action_idx >= 1:
                    last_action = trajectory[last_action_idx].get('action', '')
                    trajectory_parts.append(f"\nAction: {last_action}")
                    trajectory_parts.append(f"\nNext State:")

                trajectory_str = "".join(trajectory_parts)
                prompt = init_instruction + trajectory_str

                ground_truth = trajectory[-1].get('observation', '')
                current_observation = trajectory[-3].get('observation', '') if len(trajectory) >= 3 else ''
                last_action = trajectory[-2].get('action', '') if len(trajectory) >= 2 else ''

                metadata = {
                    'trajectory': trajectory_str,
                    'trajectory_str': trajectory_str,
                    'action': last_action,
                    'current_observation': current_observation,
                    'ground_truth_next_observation': ground_truth,
                    'is_web2nal': False
                }

                return prompt, metadata
            else:
                initial_observation = trajectory[0].get('observation', '')
                initial_action = trajectory[0].get('action', '')
                init_instruction = (
                    f"You are a web world model. I will provide you with an initial page state and a sequence of actions. "
                    f"For each action, predict the resulting page state.\n"
                    f"Strictly maintain the original format. Output only the full page state without explanations, code, or truncation.\n\n"
                )
                trajectory_str = f"Initial Page State:\n{initial_observation}\n\nFirst Action: '{initial_action}'\n\nNext Page State:"

                for i, step in enumerate(trajectory[:-1]): 
                    next_observation = step.get('next_observation', '')
                    next_action = trajectory[i+1].get('action', '')
                    trajectory_str += f"\n\n{next_observation}\n\nContinue the trajectory. Given the previous state, predict the next page state after this action.\n\nAction: '{next_action}'\n\nNext Page State:"
                prompt = init_instruction + trajectory_str
                last_step = trajectory[-1]
                metadata = {
                    'trajectory': trajectory_str,
                    'trajectory_str': trajectory_str,
                    'action': last_step.get('action'),
                    'current_observation': last_step.get('observation'),
                    'ground_truth_next_observation': last_step.get('next_observation'),
                    'is_web2nal': False  # Multi-turn trajectories are not Web2NAL tasks
                }

                return prompt, metadata
        else:
            # Extract data from the sample (existing single-step format)
            action = sample.get('action')
            current_observation = sample.get('observation')

            # Check if this is a Web2NAL task (natural language description)
            if self._is_web2nal_task():
                # For Web2NAL: task is to describe state change in natural language
                # Format matches the standard Web2NAL prompt structure
                trajectory_str = (
                    f"Page State:\n{current_observation}\n\n"
                    f"Action: {action}"
                )
                prompt = (
                    f"You are a web world model. Your task is to describe the state change in natural language. "
                    f"I will provide you with the page state and a action."
                    f"Describe the transition based on the action in the trajectory.\n\n"
                    f"Trajectory:\n{trajectory_str}\n\n"
                    f"Description:"
                )
                metadata = {
                    'trajectory': trajectory_str,
                    'trajectory_str': trajectory_str,
                    'action': action,
                    'current_observation': current_observation,
                    'ground_truth_next_observation': sample.get('next_observation'),
                    'is_web2nal': True
                }
            else:
                # For other tasks: predict the next page state
                prompt = (
                    f"You are a web world model. I will provide you with a page state and a action. "
                    f"For the action, predict the resulting page state.\n"
                    f"Strictly maintain the original format. Output only the full page state without explanations, code, or truncation.\n\n"
                    f"Page State:\n{current_observation}\n\n"
                    f"Action: '{action}'\n\n"
                    f"Next Page State:"
                )
                trajectory_str = (
                    f"Page State:\n{current_observation}\n\n"
                    f"Action: '{action}'\n\n"
                )
                metadata = {
                    'trajectory': trajectory_str,
                    'trajectory_str': trajectory_str,
                    'action': action,
                    'current_observation': current_observation,
                    'ground_truth_next_observation': sample.get('next_observation'),
                    'is_web2nal': False
                }

            return prompt, metadata

    def evaluate(self, response: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the generated text against the web world model evaluation criteria.
        
        Args:
            response: The generated text from the model (predicted next observation)
            kwargs: Optional parameters
            
        Returns:
            dict: Evaluation results, including scores for different metrics and error details
        """
        metadata = kwargs.get('metadata')
        sample_id = kwargs.get('sample_id')
        if not metadata:
            error_msg = f"Metadata for evaluation is missing for sample_id '{sample_id}'."
            print(f"[WARN] {error_msg}")
            return {
                'action_effect_accuracy_score': 0.0,
                'plausibility_score': 0.0,
                'total_score': 0.0,
                'action_effect_accuracy_reasoning': '',
                'plausibility_reasoning': '',
                'errors': [error_msg]
            }
            
        # Add the model's prediction (response) to the metadata
        metadata['predicted_next_observation'] = response
            
        errors = []
        try:
            accuracy_result = self._evaluate_accuracy_single(metadata)
        except Exception as e:
            accuracy_result = {"score": 0.0, "reasoning": ""}
            errors.append(f"Error evaluating accuracy: {str(e)}")
            
        try:
            plausibility_result = self._evaluate_plausibility_single(metadata)
        except Exception as e:
            plausibility_result = {"score": 0.0, "reasoning": ""}
            errors.append(f"Error evaluating plausibility: {str(e)}")

        scores = [accuracy_result["score"], plausibility_result["score"]]
        total_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
        
        final_scores = {
            'action_effect_accuracy_score': round(accuracy_result["score"], 4),
            'plausibility_score': round(plausibility_result["score"], 4),
            'total_score': round(total_score, 4),
            'action_effect_accuracy_reasoning': accuracy_result.get("reasoning", ""),
            'plausibility_reasoning': plausibility_result.get("reasoning", ""),
            'errors': errors
        }
        
        return final_scores

    def _robust_json_parse_all(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response and return all fields.
        
        Args:
            response: The JSON string response
            
        Returns:
            dict: All parsed fields from the JSON response
        """
        try:
            # Use json_repair to fix and parse JSON
            repaired_json = repair_json(response)
            data = json.loads(repaired_json)
            return data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to parse JSON even after repair. Error: {e}. Returning empty dict.")
        except Exception as e:
            print(f"[ERROR] Unexpected error during JSON parsing: {e}")
        
        print(f"[ERROR] Could not parse response: {response[:200]}...")
        return {}

    def _evaluate_accuracy_single(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a Web2NAL task (natural language description)
        is_web2nal = meta.get('is_web2nal', False)

        if is_web2nal:
            # For Web2NAL: evaluate natural language description accuracy
            prompt = f"""
**Role:** Natural Language Description Evaluator

**Task:** Your goal is to judge if the `predicted_description` accurately and completely describes the state change caused by the given `action`, based on the `current_page_state`. Focus on whether the description captures the key changes, new content, navigation, and UI updates.

You are provided with the `ground_truth_description` as a reference for what an ideal description should include. Compare the predicted description with the ground truth to assess accuracy and completeness.

**Input:**
<current_page_state>{meta.get('current_observation', 'N/A')}</current_page_state>
<action>{meta['action']}</action>
<predicted_description>{meta['predicted_next_observation']}</predicted_description>
<ground_truth_description>{meta.get('ground_truth_next_observation', 'N/A')}</ground_truth_description>

**Steps:**
1. Identify what changes the action should cause (navigation, content update, etc.)
2. Compare the predicted description with the ground truth: Does it capture the main changes?
3. Evaluate completeness: Are key details (page title, main content changes, new elements) mentioned?
4. Evaluate accuracy: Are the described changes correct?
5. Assign a score:
    - 1.0: Accurately describes all key state changes, highly complete
    - 0.7: Captures main changes but misses some important details
    - 0.4: Describes some changes but has significant omissions or inaccuracies
    - 0.0: Does not accurately describe the state change

**Output Format:**
Respond ONLY with a single JSON object in this format: {{"reasoning": "<your_analysis>", "action_effect_accuracy_score": <score>}}

**Your JSON Output:**
"""
        else:
            # For other tasks: evaluate web state prediction accuracy
            prompt = f"""
**Role:** Web Action Effect Evaluator

**Task:** Your goal is to judge if the `predicted_next_observation` exhibits the main and most direct expected effect of the given `action`, based on the `current_trajectory`. Focus only on whether the *core and intended causal change* triggered by the action is clearly and correctly reflected. Ignore minor differences in content, formatting, or auxiliary UI elements that do not relate to the action's main effect.

Additionally, you are provided with the `ground_truth_next_observation` as a reference for what the correct next state should look like. You can use this to better assess whether the predicted next observation captures the essential changes.

**Input:**
<current_trajectory>{meta.get('trajectory_str', 'N/A')}</current_trajectory>
<action>{meta['action']}</action>
<predicted_next_observation>{meta['predicted_next_observation']}</predicted_next_observation>
<ground_truth_next_observation>{meta.get('ground_truth_next_observation', 'N/A')}</ground_truth_next_observation>

**Steps:**
1. Identify the action and its intended main effect on the page.
2. Compare the predicted_next_observation with the current_trajectory: Did the main effect happen?
3. Reference the ground_truth_next_observation to better understand what the correct outcome should be.
4. Justify your answer briefly, referring to the main user-visible change.
5. Assign a score:
    - 1.0: Main effect present and unambiguous, ignore small differences
    - 0.7: Main effect mostly present but some clear incompleteness
    - 0.4: Action applied to right kind of element, but effect is largely wrong
    - 0.0: Main effect not present

**Output Format:**
Respond ONLY with a single JSON object in this format: {{"reasoning": "<your_analysis>", "action_effect_accuracy_score": <score>}}

**Your JSON Output:**
"""
        try:
            response = unified_call(
                backend=self.config['evaluation_model']['backend'],
                model=self.config['evaluation_model']['model'],
                prompt=prompt,
                **self.config['evaluation_model']['params']
            )
            result = self._robust_json_parse_all(response)
            score = result.get("action_effect_accuracy_score", 0.0)
            reasoning = result.get("reasoning", "")
            
            return {
                "score": max(0.0, min(1.0, float(score))),
                "reasoning": reasoning
            }
        except Exception as e:
            print(f"[ERROR] Accuracy eval LLM call failed: {e}")
            return {"score": 0.0, "reasoning": ""}
    

    def _evaluate_plausibility_single(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        options = {"A": meta['predicted_next_observation'], "B": meta['ground_truth_next_observation']}
        shuffled_keys = random.sample(list(options.keys()), 2)
        predicted_is = 'A' if shuffled_keys[0] == 'A' else 'B'

        # Check if this is a Web2NAL task (natural language description)
        is_web2nal = meta.get('is_web2nal', False)

        if is_web2nal:
            # For Web2NAL: evaluate which description is more reasonable and plausible
            prompt = f"""
**Role:** State Change Description Plausibility Evaluator.
**Task:** Given the `<current_page_state>` and the `<action>`, evaluate which of the two descriptions (`A` or `B`) **more reasonably and plausibly describes the page changes** that would result from the action.

**Evaluation Mindset:**
- A plausible description should logically match what would happen when the action is performed on the current page.
- Consider whether the described changes make sense given the action.
- Evaluate if the description accurately captures the expected state transitions and page updates.
- Check if the described outcome is realistic and consistent with typical web behavior.

**Your Analysis Steps:**
1. **Analyze A:** Does it describe a plausible and reasonable outcome for the action? What makes sense or doesn't make sense?
2. **Analyze B:** Same as above for B.
3. **Compare & Decide:** Clearly state which description more reasonably and plausibly describes the page changes that would result from the action, and why.

**Input:**
<current_page_state>\n{meta.get('current_observation', 'N/A')}</current_page_state>\n\n
<action>\n{meta['action']}</action>\n\n\n
<description_A>\n{options[shuffled_keys[0]]}</description_A>\n\n
<description_B>\n{options[shuffled_keys[1]]}</description_B>\n\n

**Output Requirements:**
1. First, provide a brief analysis of both descriptions and your reasoning.
2. Then, provide your choice (A or B).

**Output Format:**
Respond ONLY with a single JSON object in this format: {{"reasoning": "<your_analysis>", "choice": "<your_choice>"}}

**Your JSON Output:**
"""
        else:
            # For other tasks: evaluate which page state is more realistic
            prompt = f"""
**Role:** AI Turing Test Judge.
**Task:** One of the following observations (`A` or `B`) is from a real browser session, the other is generated by an AI. Based on the `<action>` taken on the `<current_trajectory>`, you must decide which observation is **more believable and realistic**.

**Evaluation Mindset:**
- A believable outcome should be a **logical and complete consequence** of the action.
- Real websites are complex. Don't be afraid to choose a complex observation if it seems more true to life.
- Pay close attention to details. Does the content make sense? Is the layout consistent? Are there any strange artifacts or nonsensical repetitions?

**Your Analysis Steps:**
1.  **Analyze A:** Is it a plausible outcome? What are its strengths and weaknesses?
2.  **Analyze B:** Same as above for B.
3. **Compare & Decide:** Clearly state which one is more likely to be the real browser observation, and why.

**Input:**
<current_trajectory>\n{meta.get('trajectory_str', 'N/A')}</current_trajectory>\n\n
<action>\n{meta['action']}</action>\n\n\n
<observation_A>\n{options[shuffled_keys[0]]}</observation_A>\n\n
<observation_B>\n{options[shuffled_keys[1]]}</observation_B>\n\n

**Output Requirements:**
1. First, provide a brief analysis of both observations and your reasoning.
2. Then, provide your choice.

**Output Format:**
Respond ONLY with a single JSON object in this format: {{"reasoning": "<your_analysis>", "choice": "<your_choice>"}}

**Your JSON Output:**
"""
        try:
            response = unified_call(
                backend=self.config['evaluation_model']['backend'],
                model=self.config['evaluation_model']['model'],
                prompt=prompt,
                **self.config['evaluation_model']['params']
            )
            result = self._robust_json_parse_all(response)
            choice = result.get("choice", "")
            reasoning = result.get("reasoning", "")
            
            score = 0.0
            if choice is not None:
                if choice == predicted_is:
                    score = 1.0
                    
            return {
                "score": score,
                "reasoning": reasoning
            }
        except Exception as e:
            print(f"[ERROR] Plausibility eval LLM call failed: {e}")
            return {"score": 0.0, "reasoning": ""}

TaskFactory.register_task('WEB_WORLD_MODEL_EVALUATION', WebWorldModelTask)