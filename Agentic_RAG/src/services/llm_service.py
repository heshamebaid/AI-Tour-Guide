import os
import json
import requests
from typing import Generator
from dotenv import load_dotenv
from pathlib import Path

# Load env from repo root (single .env for all services)
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_repo_root / ".env")


def _format_llm_error(response: requests.Response) -> str:
    """Turn OpenRouter/LLM API errors into a short, user-friendly message."""
    if response.status_code == 429:
        try:
            body = response.json()
            err = body.get("error", {})
            msg = err.get("message", "")
            meta = err.get("metadata", {}) or {}
            headers = meta.get("headers", {}) or {}
            reset_ms = headers.get("X-RateLimit-Reset")
            if reset_ms and reset_ms.isdigit():
                from datetime import datetime
                ts = int(reset_ms) / 1000
                reset_at = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
                return f"Daily free limit reached. Resets at {reset_at}. You can add credits at openrouter.ai to get more requests."
            if "free-models-per-day" in (msg or ""):
                return "Daily free limit reached. Try again tomorrow or add credits at openrouter.ai to unlock more requests."
        except Exception:
            pass
        return "Rate limit exceeded. Try again later or add credits at openrouter.ai."
    try:
        body = response.json()
        err = body.get("error", {})
        msg = err.get("message", response.text[:200] if response.text else "")
        return msg or f"LLM API error: {response.status_code}"
    except Exception:
        return f"LLM API error: {response.status_code} - {(response.text or '')[:200]}"


class SimpleLLM:
    """Simple LLM wrapper using direct API calls - no pickle issues."""
    
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
    
    def invoke(self, prompt: str) -> 'SimpleResponse':
        """Call the LLM API directly."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.ok:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return SimpleResponse(content)
        else:
            raise Exception(_format_llm_error(response))
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream the LLM response token by token."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60,
            stream=True
        )
        
        if not response.ok:
            raise Exception(_format_llm_error(response))
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue


class SimpleResponse:
    """Simple response object to match LangChain interface."""
    def __init__(self, content: str):
        self.content = content


class LLMService:
    """LLM Service using OpenRouter only (no fallback)."""
    
    def __init__(self):
        _raw = os.getenv("OPEN_ROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or "liquid/lfm-2.5-1.2b-thinking:free"
        # Qwen free model has no endpoints; use Liquid instead
        self._model = "liquid/lfm-2.5-1.2b-thinking:free" if _raw and "qwen" in _raw.lower() else (_raw or "liquid/lfm-2.5-1.2b-thinking:free")
        self._api_key = os.getenv("OPEN_ROUTER_API_KEY", "")
        self._base_url = "https://openrouter.ai/api/v1"
        self._llm = None
    
    @property
    def llm(self) -> SimpleLLM:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = SimpleLLM(self._model, self._api_key, self._base_url)
        return self._llm
