"""OpenAI-compatible client for policy inference services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLLMClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "mm-webagent"

    def query(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use VLLMClient.") from exc

        client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
