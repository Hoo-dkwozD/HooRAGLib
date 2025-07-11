#!/usr/bin/env python3

"""
Base class for LLM wrapper classes in HooRAGLib.
This class defines the interface for LLMs that can be used with Retrieval objects.

:author: Hoo-dkwozD
:version: 1.0.0
:date: 2025-07-11
"""

# Python Standard Library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Third-party imports

# Local imports

type ModelResponse = Dict[str, Any]


class BaseLLM(ABC):
    """
    Defines the interface for LLM Wrappers that can be used with Retrieval objects.
    """

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the BaseLLM with a model name and optional parameters.
        
        :param model_name: The name of the LLM model to use.
        :param kwargs: Additional parameters for the LLM.
        """
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
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
        Generate a response from the LLM based on the provided prompt.
        
        :param prompt: The input prompt to generate a response for.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: Sampling temperature for generation.
        :param top_p: Top-p sampling parameter.
        :param stop: Optional list of stop sequences.
        :param kwargs: Additional parameters for the LLM.
        :return: A dictionary containing the generated response.
        """

        pass
