import logging
import json
import re
from typing import List, Union, Dict, Any
from pydantic import BaseModel, ValidationError
import anthropic
import os
import openai
import dotenv
import requests

dotenv.load_dotenv()

log = logging.getLogger(__name__)

class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")

# Base LLM class to handle common functionality
class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None

    def _extract_json(self, response_text: str) -> str:
        """Extract JSON from response text that may contain extra content.

        Handles cases like:
        - Text wrapped in <think>...</think> tags
        - JSON in markdown code blocks
        - Plain JSON with surrounding text
        """
        # First, try to find JSON in markdown code blocks
        markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if markdown_match:
            return markdown_match.group(1)

        # Try to extract JSON between first { and last }
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return response_text[first_brace:last_brace + 1]

        # Return as-is if no extraction patterns match
        return response_text

    def _filter_invalid_enums(self, data: Dict[str, Any], response_model: BaseModel) -> Dict[str, Any]:
        """Filter out invalid enum values from vulnerability_types field.

        This prevents validation errors when LLM returns vulnerability types
        not in the defined enum (e.g., 'DoS', 'CSRF', etc.).
        """
        if 'vulnerability_types' not in data:
            return data

        # Get the enum field from the response model
        vuln_types_field = response_model.model_fields.get('vulnerability_types')
        if not vuln_types_field:
            return data

        # Extract the enum class from the List[Enum] type annotation
        # Handle List[EnumType] annotation
        field_annotation = vuln_types_field.annotation
        enum_class = None

        # Try to extract enum from typing.List[EnumClass]
        if hasattr(field_annotation, '__args__'):
            enum_class = field_annotation.__args__[0]

        if not enum_class or not hasattr(enum_class, '__members__'):
            return data

        # Get valid enum values
        valid_values = set(enum_class.__members__.keys())

        # Filter vulnerability_types list
        original_types = data['vulnerability_types']
        filtered_types = [vt for vt in original_types if vt in valid_values]

        # Log if any were filtered out
        if len(filtered_types) < len(original_types):
            invalid_types = set(original_types) - set(filtered_types)
            log.warning("Filtered out invalid vulnerability types: %s", invalid_types)

        data['vulnerability_types'] = filtered_types
        return data

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        try:
            # Apply prefill if set
            if self.prefill:
                response_text = self.prefill + response_text

            # Extract JSON from potentially wrapped response
            json_text = self._extract_json(response_text)

            # Parse JSON to dict for enum filtering
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as json_err:
                log.warning("Failed to parse JSON: %s", json_err)
                raise LLMError("Validation failed") from json_err

            # Filter invalid enum values
            data = self._filter_invalid_enums(data, response_model)

            # Convert back to JSON and validate with pydantic
            filtered_json = json.dumps(data)
            return response_model.model_validate_json(filtered_json)

        except ValidationError as e:
            log.warning("[-] Response validation failed\n", exc_info=e)
            raise LLMError("Validation failed") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Dict[str, Any]) -> None:
        usage_info = response.usage.__dict__
        log.debug("Received chat response", extra={"usage": usage_info})

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 4096) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        self._log_response(response)

        response_text = self.get_response(response)
        if response_model:
            response_text = self._validate_response(response_text, response_model) if response_model else response_text
        self._add_to_history("assistant", response_text)
        return response_text

class Claude(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        # API key is retrieved from an environment variable by default
        self.client = anthropic.Anthropic(max_retries=3, base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{    \"scratchpad\": \"1."
            messages = [{"role": "user", "content": user_prompt}, 
                        {"role": "assistant", "content": self.prefill}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            # response_model is not used here, only in ChatGPT
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text.replace('\n', '')


class ChatGPT(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model=None) -> Dict[str, Any]:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            # Add response format configuration if a model is provided
            if response_model:
                params["response_format"] = {
                    "type": "json_object"
                }

            return self.client.chat.completions.create(**params)
        except openai.APIConnectionError as e:
            raise APIConnectionError("The server could not be reached") from e
        except openai.RateLimitError as e:
            raise RateLimitError("Request was rate-limited; consider backing off") from e
        except openai.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred: {str(e)}") from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        return response


class Ollama(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model

    def create_messages(self, user_prompt: str) -> str:
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "options": {
            "temperature": 1,
            "system": self.system_prompt,
            }
            ,"stream":False,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            return response
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif e.response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(e.response.status_code, e.response.json()) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.json()['response']
        return response

    def _log_response(self, response: Dict[str, Any]) -> None:
        log.debug("Received chat response", extra={"usage": "Ollama"})

