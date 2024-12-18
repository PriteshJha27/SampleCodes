import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("openai_api_key")
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import requests


# Use it in a simple chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

class CustomOpenAILLM(LLM):
    """Custom LLM wrapper for OpenAI-compatible API."""
    
    # Model configuration
    api_url: str = Field(..., description="URL for the API endpoint")
    api_key: Optional[str] = Field(None, description="API key if required")
    model_kwargs: Dict = Field(default_factory=dict, description="Additional parameters to pass to the model")
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api_url exists."""
        api_url = values.get("api_url")
        if not api_url:
            raise ValueError("api_url must be provided")
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_openai"

    def _prepare_request_payload(self, prompt: str) -> Dict:
        """Prepare the payload for the API request."""
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **self.model_kwargs
        }
        return payload

    def _prepare_headers(self) -> Dict:
        """Prepare headers for the API request."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the API call to the custom endpoint.

        Args:
            prompt: The prompt to send to the model.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for monitoring the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the API call fails.
        """
        headers = self._prepare_headers()
        payload = self._prepare_request_payload(prompt)
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse the response - adjust this based on your API's response format
            response_data = response.json()
            
            # Assuming the response format is similar to OpenAI's
            # Modify this based on your actual API response structure
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]
            else:
                return response_data.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call failed: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_url": self.api_url,
            "model_kwargs": self.model_kwargs
        }
llm = CustomOpenAILLM(
    api_url="https://api.openai.com/v1/chat/completions",
    api_key=api_key,
    model_kwargs={
        "model": "gpt-4o-mini",  # or your specific model name
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "programming"})
result
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import requests

class LlamaLLM(LLM):
    """Custom LLM wrapper for Llama API."""
    
    api_url: str = Field(..., description="URL for the Llama API endpoint")
    api_key: Optional[str] = Field(None, description="API key if required")
    max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api_url exists."""
        api_url = values.get("api_url")
        if not api_url:
            raise ValueError("api_url must be provided")
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llama"

    def _prepare_request_payload(self, prompt: str) -> Dict:
        """Prepare the payload for the Llama API request."""
        return {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False  # Set to True if you want to implement streaming
        }

    def _prepare_headers(self) -> Dict:
        """Prepare headers for the API request."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the API call to the Llama endpoint."""
        headers = self._prepare_headers()
        payload = self._prepare_request_payload(prompt)
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse the Llama API response
            # Modify this based on your actual API response structure
            response_data = response.json()
            
            # Assuming the response has a 'generated_text' or similar field
            # Adjust this based on your actual API response structure
            if "generated_text" in response_data:
                return response_data["generated_text"]
            elif "response" in response_data:
                return response_data["response"]
            elif "output" in response_data:
                return response_data["output"]
            else:
                raise ValueError(f"Unexpected response format: {response_data}")
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call failed: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_url": self.api_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
