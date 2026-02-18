from openai import OpenAI
import time
import random
import re

MODEL_TO_API_BASE = {
    "Qwen3-8B": "http://x.xx.xx.x:8012/v1",
    "WebWorld-8B": "http://x.xx.xx.x:8012/v1",
}

def get_api_base_from_model(model_name):
    """
    Dynamically determine the openai_api_base based on the model name.
    :param model_name: Model name, e.g., "Qwen2___5-3B-Instruct"
    :return: The corresponding openai_api_base address
    """
    try:
        api_base = MODEL_TO_API_BASE.get(model_name)
        if not api_base:
            raise ValueError(f"No API address found for model {model_name}")
        # If there are multiple addresses, randomly select one
        if isinstance(api_base, list):
            return random.choice(api_base)
        return api_base
    except Exception as e:
        raise ValueError(f"Unable to resolve model name: {model_name}. Error: {str(e)}")


def call_api(model, messages, temperature=0.1, max_retries=30, **kwargs):
    """
    Call the API with support for dynamically setting the base_url.
    Automatically removes <reason> tag content from the response.
    """
    # Dynamically determine the openai_api_base
    openai_api_base = get_api_base_from_model(model)
    openai_api_key = "EMPTY"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    kwargs['stream'] = False
    
    for _ in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            if kwargs.get("stream") == False:
                raw_response = chat_response.choices[0].message.content.strip()
            else:
                full_response = ""
                for chunk in chat_response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                raw_response = full_response
            
            return raw_response
            
        except Exception as e:
            print(f"API error: {str(e)}, retrying...")
            time.sleep(1)
    raise Exception("Max retries reached")

if __name__ == "__main__":
    # Actual API call test
    print("Test case 3: Actual API call")
    model = "qwen3-8b"
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    response = call_api(model, messages, temperature=0.7, stream=False)
    print(f"API response: {response}")
