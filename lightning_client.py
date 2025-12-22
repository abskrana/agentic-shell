"""
Lightning AI client for unified backend services.
Provides a client interface for communicating with the Lightning AI backend
for AI text generation and speech-to-text transcription services.
"""

import aiohttp
from typing import Optional, Dict, Any, List


class LightningAIClient:
    """Client for Lightning AI unified backend communication."""
    
    def __init__(self, base_url: str):
        """Initialize Lightning AI client.
        
        Args:
            base_url: Base URL of Lightning AI backend (e.g., https://abc123.lightning.ai).
        """
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for API requests.
        
        Returns:
            Dictionary of HTTP headers.
        """
        return {'Content-Type': 'application/json'}
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists and is not closed."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def generate_content(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        model: Optional[str] = None
    ) -> str:
        """Generate AI response using the Lightning AI backend.
        
        Args:
            prompt: The prompt to send to the AI model.
            temperature: Temperature for generation (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            model: Model to use ('gemini' or 'qwen'), if None uses backend default.
            
        Returns:
            Generated text response.
            
        Raises:
            Exception: If the request fails.
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/api/ai/generate"
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if model:
            payload["model"] = model
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('text', '')
                else:
                    error_text = await response.text()
                    raise Exception(f"Lightning AI returned status {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Failed to connect to Lightning AI: {str(e)}")
    
    async def transcribe_audio(
        self,
        audio_content: str,
        language_code: str = "en-US",
        sample_rate: int = 48000,
        context_phrases: Optional[List[str]] = None,
        context_boost: float = 15.0
    ) -> Dict[str, Any]:
        """Transcribe audio using Lightning AI Speech-to-Text.
        
        Args:
            audio_content: Base64 encoded audio data.
            language_code: Language code (e.g., "en-US", "hi-IN").
            sample_rate: Audio sample rate in Hz.
            context_phrases: List of phrases for speech context to improve accuracy.
            context_boost: Boost value for context phrases (0.0 to 20.0).
            
        Returns:
            Dictionary with transcript and confidence information.
            
        Raises:
            Exception: If the transcription request fails.
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/api/speech/transcribe"
        payload = {
            "audio_content": audio_content,
            "language_code": language_code,
            "sample_rate": sample_rate,
            "context_phrases": context_phrases,
            "context_boost": context_boost
        }
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Speech transcription failed: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    async def get_config(self) -> Dict[str, Any]:
        """Get backend configuration.
        
        Returns:
            Configuration dictionary from the Lightning AI backend.
            
        Raises:
            Exception: If the request fails.
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/api/config"
        
        try:
            async with self.session.get(
                url,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get config: {response.status}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Failed to connect: {str(e)}")


class LightningAIModel:
    """Model wrapper compatible with generative AI interfaces.
    
    Provides compatibility with standard AI model interfaces while using
    the Lightning AI backend for actual content generation.
    """
    
    def __init__(self, base_url: str):
        """Initialize the Lightning AI model wrapper.
        
        Args:
            base_url: Base URL of the Lightning AI backend.
        """
        self.client = LightningAIClient(base_url)
        self._config = None
        self._current_model = None
    
    def set_model(self, model: str) -> None:
        """Set the current model to use.
        
        Args:
            model: Model name ('gemini' or 'qwen').
        """
        self._current_model = model
    
    async def _load_config(self) -> None:
        """Load backend configuration if not already loaded."""
        if self._config is None:
            self._config = await self.client.get_config()
    
    async def generate_content_async(
        self,
        prompt: str,
        generation_config: Optional[Dict] = None,
        model: Optional[str] = None
    ):
        """Generate content asynchronously.
        
        Args:
            prompt: The prompt to send to the AI model.
            generation_config: Optional configuration with 'temperature' and 'max_output_tokens'.
            model: Model to use, overrides instance model if provided.
            
        Returns:
            Response object containing the generated text.
        """
        temperature = 1.0
        max_tokens = 2048
        
        if generation_config:
            temperature = generation_config.get('temperature', 1.0)
            max_tokens = generation_config.get('max_output_tokens', 2048)
        
        # Use provided model, or fall back to instance model, or None (backend default)
        selected_model = model or self._current_model
        
        text = await self.client.generate_content(
            prompt, temperature, max_tokens, selected_model
        )
        return LightningAIResponse(text)
    
    async def close(self) -> None:
        """Close the client session."""
        await self.client.close()


class LightningAIResponse:
    """Response wrapper compatible with generative AI response interfaces."""
    
    def __init__(self, text: str):
        """Initialize the response with generated text.
        
        Args:
            text: The generated text content.
        """
        self._text = text
    
    @property
    def text(self) -> str:
        """Get the generated text.
        
        Returns:
            The generated text content.
        """
        return self._text
