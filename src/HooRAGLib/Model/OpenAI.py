#!/usr/bin/env python3

"""
Wrapper class for OpenAI models that can interface with Retrieval objects in HooRAGLib.
This class extends BaseModel and provides methods to interact with OpenAI's API.

:author: Hoo-dkwozD
:version: 1.0.0
:date: 2025-07-11
"""

# Python Standard Library imports
import os
from typing import Any, Dict, List, Optional

# Third-party imports
import openai

# Local imports
from HooRAGLib.Model.BaseLLM import BaseLLM
from HooRAGLib.Model.BaseLLM import ModelResponse


class OpenAI(BaseLLM):
    """
    Wrapper class that interacts with OpenAI API.
    Inherits from BaseModel.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the OpenAI wrapper with a model name and optional parameters.
        
        :param model_name: The name of the OpenAI model to use.
        :param kwargs: Additional parameters for the OpenAI API.
        """

        openai.api_key = os.getenv("OPENAI_API_KEY", kwargs.get("api_key", ""))
        if not openai.api_key:
            raise ValueError("OpenAI API key is required. Set it in the environment variable OPENAI_API_KEY or pass it as a parameter.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response from the OpenAI model based on the provided prompt.
        
        :param prompt: The input prompt to generate a response for.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: Sampling temperature for generation.
        :param top_p: Top-p sampling parameter.
        :param stop: Optional list of stop sequences.
        :param kwargs: Additional parameters for the OpenAI API.
        :return: A dictionary containing the generated response.
        """

        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )

        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [choice.text for choice in response.choices],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
