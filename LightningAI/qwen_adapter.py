"""
Qwen Adapter for Lightning Studio Backend
==========================================
This adapter provides a Gemini-compatible interface for Qwen models deployed with vLLM.
It's designed to work seamlessly with the Lightning AI Unified Backend.

Usage in Lightning Studio:
    from qwen_adapter import GenerativeModel
    model = GenerativeModel(model_name="your-model-name")
    response = model.generate_content("Your prompt here")
    print(response.text)

Environment Variables Required:
    - QWEN_API_URL: The vLLM API endpoint (e.g., http://localhost:8000)
    - QWEN_MODEL_NAME: The model identifier (optional, auto-detected from /v1/models)

Compatible with both sync and async operations.
"""

import os
import aiohttp
import requests
import json
from typing import List, Dict, Union, Optional, Any


# ========================================
# Mock Gemini Response Objects
# ========================================

class QwenResponsePart:
    """Mimics Gemini's response part structure"""
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text


class QwenResponse:
    """Mimics Gemini's response structure with text and parts"""
    def __init__(self, text: str):
        self.parts = [QwenResponsePart(text)]
        self.text = text
        self._result = text
    
    @classmethod
    def from_error(cls, error_msg: str):
        """Create an error response"""
        print(f"⚠️ QwenAdapter Error: {error_msg}")
        return cls(f"Error: {error_msg}")
    
    def __str__(self):
        return self.text


# ========================================
# Generation Configuration
# ========================================

class GenerationConfig:
    """Configuration for text generation (Gemini-compatible)"""
    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ):
        self.temperature = temperature if temperature is not None else 0.1
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else 2048
        self.top_p = top_p if top_p is not None else 0.95
        self.top_k = top_k if top_k is not None else 40
        self.stop_sequences = stop_sequences if stop_sequences else ["<|im_end|>", "\n\n\n"]


# ========================================
# Qwen Adapter (Main Class)
# ========================================

class QwenAdapter:
    """
    Adapter that provides Gemini-like interface for Qwen models via vLLM.
    Supports both synchronous and asynchronous generation.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Qwen adapter
        
        Args:
            model_name: Name of the model to use. If None, auto-detects from vLLM /v1/models endpoint
        """
        self.api_url = os.getenv("QWEN_API_URL")
        if not self.api_url:
            raise ValueError(
                "QWEN_API_URL environment variable is not set. "
                "Please set it to your vLLM endpoint (e.g., http://localhost:8000)"
            )
        
        self.api_url = self.api_url.rstrip('/')
        self.api_endpoint = f"{self.api_url}/v1/chat/completions"
        self.models_endpoint = f"{self.api_url}/v1/models"
        
        # Auto-detect model name if not provided
        if model_name is None:
            model_name = self._auto_detect_model()
        
        self.model_name = model_name
        
        # Default generation config
        self.default_config = GenerationConfig()
        
        print(f"🤖 QwenAdapter initialized")
        print(f"   Endpoint: {self.api_endpoint}")
        print(f"   Model: {self.model_name}")
    
    def _auto_detect_model(self) -> str:
        """
        Auto-detect the model name from vLLM's /v1/models endpoint
        Returns the first available model
        """
        try:
            response = requests.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0]["id"]
                    print(f"✓ Auto-detected model: {model_id}")
                    return model_id
        except Exception as e:
            print(f"⚠️ Could not auto-detect model: {e}")
        
        # Fallback to environment variable or default
        fallback = os.getenv("QWEN_MODEL_NAME", "qwen-model")
        print(f"ℹ️  Using fallback model name: {fallback}")
        return fallback
    
    def _parse_prompt(self, prompt: Union[str, List, Dict]) -> str:
        """
        Parse various prompt formats (Gemini-compatible)
        
        Args:
            prompt: Can be:
                - String: "Your prompt here"
                - List: [{"parts": ["text1", "text2"]}]
                - Dict: {"parts": ["text1", "text2"]}
        
        Returns:
            Parsed prompt string
        """
        if isinstance(prompt, str):
            return prompt
        
        elif isinstance(prompt, list):
            try:
                # Extract parts from list format
                parts = []
                for item in prompt:
                    if isinstance(item, dict) and "parts" in item:
                        parts.extend(item["parts"])
                    elif isinstance(item, str):
                        parts.append(item)
                return " ".join(str(p) for p in parts)
            except Exception as e:
                raise ValueError(f"Could not parse list prompt format: {e}")
        
        elif isinstance(prompt, dict):
            try:
                # Extract parts from dict format
                parts = prompt.get("parts", [])
                return " ".join(str(p) for p in parts)
            except Exception as e:
                raise ValueError(f"Could not parse dict prompt format: {e}")
        
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    
    def _build_payload(
        self,
        user_content: str,
        generation_config: Optional[Union[GenerationConfig, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Build the API payload for vLLM
        
        Args:
            user_content: The user's prompt
            generation_config: Generation configuration
        
        Returns:
            API payload dictionary
        """
        # Use provided config or default
        if generation_config is None:
            config = self.default_config
        elif isinstance(generation_config, dict):
            config = GenerationConfig(**generation_config)
        else:
            config = generation_config
        
        # Build the payload matching vLLM's chat completions format
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful bash command assistant. Convert natural language instructions to bash commands."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop": config.stop_sequences
        }
        
        return payload
    
    def generate_content(
        self,
        prompt: Union[str, List, Dict],
        generation_config: Optional[Union[GenerationConfig, Dict]] = None
    ) -> QwenResponse:
        """
        Generate content synchronously (Gemini-compatible interface)
        
        Args:
            prompt: The prompt in various formats (string, list, dict)
            generation_config: Optional generation configuration
        
        Returns:
            QwenResponse object with .text attribute
        """
        try:
            # Parse prompt
            user_content = self._parse_prompt(prompt)
            
            # Build payload
            payload = self._build_payload(user_content, generation_config)
            
            # Make synchronous API call
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data["choices"][0]["message"]["content"].strip()
                return QwenResponse(generated_text)
            else:
                error_text = response.text
                return QwenResponse.from_error(
                    f"API returned status {response.status_code}: {error_text}"
                )
        
        except requests.exceptions.Timeout:
            return QwenResponse.from_error("Request timeout after 30 seconds")
        except requests.exceptions.ConnectionError:
            return QwenResponse.from_error(f"Could not connect to {self.api_url}")
        except Exception as e:
            return QwenResponse.from_error(f"An exception occurred: {e}")
    
    async def generate_content_async(
        self,
        prompt: Union[str, List, Dict],
        generation_config: Optional[Union[GenerationConfig, Dict]] = None
    ) -> QwenResponse:
        """
        Generate content asynchronously (Gemini-compatible interface)
        
        Args:
            prompt: The prompt in various formats (string, list, dict)
            generation_config: Optional generation configuration
        
        Returns:
            QwenResponse object with .text attribute
        """
        try:
            # Parse prompt
            user_content = self._parse_prompt(prompt)
            
            # Build payload
            payload = self._build_payload(user_content, generation_config)
            
            # Make asynchronous API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        generated_text = data["choices"][0]["message"]["content"].strip()
                        return QwenResponse(generated_text)
                    else:
                        error_text = await response.text()
                        return QwenResponse.from_error(
                            f"API returned status {response.status}: {error_text}"
                        )
        
        except aiohttp.ClientError as e:
            return QwenResponse.from_error(f"Network error: {e}")
        except Exception as e:
            return QwenResponse.from_error(f"An exception occurred: {e}")


# ========================================
# Gemini-Compatible Factory Function
# ========================================

def GenerativeModel(model_name: Optional[str] = None) -> QwenAdapter:
    """
    Factory function that mimics google.generativeai.GenerativeModel
    
    Args:
        model_name: Name of the model to use (optional, auto-detects if None)
    
    Returns:
        QwenAdapter instance
    
    Example:
        model = GenerativeModel("qwen-model")
        response = model.generate_content("Hello, how are you?")
        print(response.text)
    """
    return QwenAdapter(model_name=model_name)