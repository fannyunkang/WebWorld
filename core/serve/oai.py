import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'  # Replace with your OpenAI API Key

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)


def call_api(model, messages, max_retries=30, **kwargs):
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = 8192

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}, retrying...")

    raise Exception("Max retries failed")


if __name__ == "__main__":
    print("Testing OpenAI official API call")

    test_messages = [
        {"role": "user", "content": "Hello, this is a test message. Please reply in one sentence."}
    ]

    try:
        response = call_api("gpt-4o-2024-11-20", test_messages, max_retries=3)
        print("API call succeeded!")
        print("Response:", response)
    except Exception as e:
        print(f"API call failed: {e}")
