import os
import json
import requests
from typing import Generator
from dotenv import load_dotenv
from pathlib import Path

# Load env directly
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(str(env_path))


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
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
    
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
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
        
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
    """LLM Service using direct API calls - fully picklable."""
    
    def __init__(self):
        self._model = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._base_url = "https://openrouter.ai/api/v1"
        self._llm = None
    
    @property
    def llm(self) -> SimpleLLM:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = SimpleLLM(self._model, self._api_key, self._base_url)
        return self._llm
