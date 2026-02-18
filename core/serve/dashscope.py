import os
import re
import time
import traceback
from openai import OpenAI

os.environ['DASHSCOPE_API_KEY'] = ''

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def call_api(model, messages, seed=42, max_retries=30, **kwargs):
    for _ in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
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
            print(f"Error: {str(e)}, retrying...")
            traceback.print_exc()
            time.sleep(1)
    raise Exception("Max retries exceeded")

if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Hello, this is a test message."}
    ]

    try:
        response = call_api(
            model="qwen-max",
            messages=test_messages,
            stream=False
        )
        print("API call successful:")
        print(response)
    except Exception as e:
        print(f"API call failed: {str(e)}")
        traceback.print_exc()
