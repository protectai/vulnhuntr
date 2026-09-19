import os
import re
import json
import logging
from typing import List, Union, Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError
import anthropic
import openai
from google import genai
from google.genai import types
import dotenv
import requests

dotenv.load_dotenv()

log = logging.getLogger("vulnhuntr.llm")


class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass


class RateLimitError(LLMError):
    pass


class APIConnectionError(LLMError):
    pass


class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Any):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")


class UsageStats:
    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_calls: int = 0

    def record(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_calls += 1

    def summary(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def extract_clean_json(text: str) -> str:
    """
    Extracts valid JSON string from raw model output, stripping markdown code fences
    or surrounding commentary if present.
    """
    cleaned = text.strip()
    # Match markdown code block ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", cleaned, re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()
    
    # If the text still does not start with { or [, locate the first { and last }
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            cleaned = cleaned[start_brace : end_brace + 1]

    return cleaned


# Base LLM class to handle common functionality
class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None
        self.usage = UsageStats()

    def _validate_response(self, response_text: str, response_model: Type[BaseModel]) -> BaseModel:
        if self.prefill and not response_text.strip().startswith("{"):
            response_text = self.prefill + response_text

        cleaned_text = extract_clean_json(response_text)
        try:
            return response_model.model_validate_json(cleaned_text)
        except ValidationError as e:
            # Fallback: try parsing as dict with loose python json then validate
            try:
                data = json.loads(cleaned_text)
                return response_model.model_validate(data)
            except Exception:
                log.warning("[-] Response validation failed on response text:\n%s", response_text, exc_info=e)
                raise LLMError("Validation failed") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Any) -> None:
        pass

    def chat(self, user_prompt: str, response_model: Optional[Type[BaseModel]] = None, max_tokens: int = 4096) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        self._log_response(response)

        response_text = self.get_response(response)
        if response_model:
            parsed = self._validate_response(response_text, response_model)
            self._add_to_history("assistant", response_text)
            return parsed
        
        self._add_to_history("assistant", response_text)
        return response_text


class Gemini(LLM):
    """Google Gemini LLM provider using modern google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-pro", api_key: Optional[str] = None, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=resolved_key)
        self.model = model

    def create_messages(self, user_prompt: str) -> str:
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int, response_model: Optional[Type[BaseModel]] = None) -> Any:
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt if self.system_prompt else None,
                max_output_tokens=max_tokens,
                temperature=0.2,
            )
            if response_model:
                config.response_mime_type = "application/json"
                config.response_schema = response_model

            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            return response
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "resource_exhausted" in error_str:
                raise RateLimitError("Gemini rate limit exceeded") from e
            if "connection" in error_str:
                raise APIConnectionError("Failed to connect to Google Gemini API") from e
            raise LLMError(f"Gemini API error: {e}") from e

    def get_response(self, response: Any) -> str:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            c_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            self.usage.record(p_tokens, c_tokens)
        return response.text or ""


class Claude(LLM):
    def __init__(self, model: str, base_url: str = "https://api.anthropic.com", system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{\n  \"scratchpad\": \"1."
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.prefill}
            ]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: Optional[Type[BaseModel]] = None) -> Any:
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages,
                temperature=0.2,
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Anthropic server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Anthropic request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Any) -> str:
        if hasattr(response, "usage") and response.usage:
            self.usage.record(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text


class ChatGPT(LLM):
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: Optional[Type[BaseModel]] = None) -> Any:
        try:
            params: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }

            if response_model:
                params["response_format"] = {"type": "json_object"}

            return self.client.chat.completions.create(**params)
        except openai.APIConnectionError as e:
            raise APIConnectionError("The OpenAI server could not be reached") from e
        except openai.RateLimitError as e:
            raise RateLimitError("OpenAI request was rate-limited") from e
        except openai.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e
        except Exception as e:
            raise LLMError(f"An unexpected OpenAI error occurred: {str(e)}") from e

    def get_response(self, response: Any) -> str:
        if hasattr(response, "usage") and response.usage:
            self.usage.record(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content or ""


class Ollama(LLM):
    """Ollama local model runner via OpenAI compatible endpoint or direct HTTP."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/api/generate", system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model
        # Check if user provided an OpenAI-compatible /v1 endpoint
        self.is_openai_compat = "/v1" in base_url
        if self.is_openai_compat:
            self.openai_client = openai.OpenAI(base_url=base_url, api_key="ollama")

    def create_messages(self, user_prompt: str) -> Any:
        if self.is_openai_compat:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return messages
        return user_prompt

    def send_message(self, user_prompt_or_messages: Any, max_tokens: int, response_model: Optional[Type[BaseModel]] = None) -> Any:
        if self.is_openai_compat:
            return self.openai_client.chat.completions.create(
                model=self.model,
                messages=user_prompt_or_messages,
                response_format={"type": "json_object"} if response_model else None,
                temperature=0.2,
            )

        payload = {
            "model": self.model,
            "prompt": user_prompt_or_messages,
            "format": "json" if response_model else "",
            "options": {
                "temperature": 0.2,
                "system": self.system_prompt,
            },
            "stream": False,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            if response.status_code == 429:
                raise RateLimitError("Ollama request was rate-limited")
            elif response.status_code >= 500:
                raise APIConnectionError("Ollama server error")
            elif response.status_code != 200:
                raise APIStatusError(response.status_code, response.text)
            return response
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Failed to reach Ollama: {e}") from e

    def get_response(self, response: Any) -> str:
        if self.is_openai_compat:
            return response.choices[0].message.content or ""
        return response.json().get("response", "")
