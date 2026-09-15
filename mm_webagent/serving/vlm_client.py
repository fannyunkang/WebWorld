"""OpenAI-compatible vision-language client for Qwen-VL style inference."""

from __future__ import annotations

from dataclasses import dataclass

from mm_webagent.data.screenshot_encoder import encode_screenshot


@dataclass
class VLMClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "mm-webagent"
    api_key: str = "EMPTY"
    temperature: float = 0.0

    def query(self, prompt: str, screenshot: str | None = None, screenshot_path: str | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use VLMClient.") from exc

        image_path = screenshot or screenshot_path
        content: str | list[dict] = prompt
        if image_path:
            payload = encode_screenshot(image_path)
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{payload.mime_type};base64,{payload.base64_data}",
                    },
                },
            ]

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
