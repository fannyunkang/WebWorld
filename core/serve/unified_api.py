import argparse
import os
import sys
from pathlib import Path
import re

# Add the project root directory to Python path so we can import modules
# This allows the script to run independently from different locations
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import with standard production paths first
try:
    from core.serve.dashscope import call_api as dashscope_call
    from core.serve.dlc import call_api as dlc_call
    from core.serve.huggingface import call_api as huggingface_call
    from core.serve.oai import call_api as oai_call
    IMPORT_SUCCESS = True
except ImportError:
    # If that fails, try direct imports (for when running from the script's directory)
    try:
        import dashscope
        import dlc
        import huggingface
        import oai
        from dashscope import call_api as dashscope_call
        from dlc import call_api as dlc_call
        from huggingface import call_api as huggingface_call
        from oai import call_api as oai_call
        IMPORT_SUCCESS = True
    except ImportError:
        # If all imports fail, set flag to False
        IMPORT_SUCCESS = False
        print("Warning: Could not import API modules. Functions will not work until proper modules are installed.")

def remove_reason_tags(text):
    """
    Remove <reason>...</reason> tags and their content from the text.
    If the tags do not exist, return the original text.
    
    :param text: Original text
    :return: Processed text
    """
    # Use regex to match <reason>...</reason> (supports multiline and nested content)
    pattern = r'<reason>.*?</reason>\s*'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def unified_call(backend, model, prompt, parse_reason=True, **kwargs):
    """
    Unified interface for calling different LLM backends.

    Args:
        backend (str): The backend to use.
        model (str): The model identifier.
        prompt (str): The input prompt.
        parse_reason (bool): Whether to strip <reason>...</reason> tags 
                             from the response. Defaults to True.
        **kwargs: Additional keyword arguments (e.g., temperature, max_tokens).

    Returns:
        str: The model's response, optionally cleaned of reason tags.
    """
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    # ---- Dispatch to the appropriate backend ----
    if backend == 'dashscope':
        raw_response = dashscope_call(model=model, messages=messages, **kwargs)
    elif backend == 'dlc':
        raw_response = dlc_call(model=model, messages=messages, **kwargs)
    elif backend == 'oai':
        raw_response = oai_call(model=model, messages=messages, **kwargs)
    elif backend == 'openai':
        raw_response = openai_call(model=model, messages=messages, **kwargs)
    elif backend == 'huggingface':
        raw_response = huggingface_call(model=model, messages=messages, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # ---- Optionally clean <reason>...</reason> tags ----
    if parse_reason:
        return remove_reason_tags(raw_response)

    return raw_response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Model Calling Interface')
    parser.add_argument('--backend', type=str, default='oai',
                      choices=['dashscope', 'dlc', 'openai', 'oai', 'huggingface'],
                      help='Select the backend service to use (default: oai)')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20',
                      help='Name of the model to use (default: gpt-4o-2024-11-20)')
    parser.add_argument('--prompt', type=str, default='Hello',
                      help='User input prompt (default: Hello)')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Generation temperature parameter (default: 0.1)')
    parser.add_argument('--max_tokens', type=int, default=1024*8,
                      help='Maximum number of generated tokens (default: 8192)')

    args = parser.parse_args()

    prompt = args.prompt
    try:
        response = unified_call(
            backend="oai",
            model="gpt-4o-2024-11-20",
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"[{args.backend.upper()}] Response:\n{response}")
    except Exception as e:
        print(f"Call failed: {str(e)}")
