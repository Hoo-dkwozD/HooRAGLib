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
from HooRAGLib.Retrievers.BaseRetriever import BaseRetriever


type WrapperResponse = Dict[str, Any]


class BaseLLM(ABC):
    """
    Defines the interface for LLM Wrappers that can be used with Retrieval objects.
    """

    client: Any
    models: list[str]
    model_name: str
    model_version: Optional[str]
    system_prompt: Optional[str]
    retriever: Optional[BaseRetriever]

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the LLM wrapper with a model name and optional parameters.
        
        :param model_name: The name of the LLM model wrapper instance.
        :param kwargs: Additional parameters for the LLM API.

        :raises ValueError: If the api key is not provided.
        :raises AuthenticationError: If the provided API key is invalid.
        """

        pass

    @abstractmethod
    def configure(
        self, 
        model_version: str,
        system_prompt: Optional[str],
        retriever: Optional[BaseRetriever],
    ) -> WrapperResponse:
        """
        Configure the LLM Wrapper with necessary settings & components.
        
        :param model_version: The version of the LLM model to use.
        :param retriever: An optional BaseRetriever instance for RAG.

        :return: A dictionary containing the configuration response.

        :raises ValueError: If the model version is not specified or invalid.
        :raises TypeError: If the retriever is not an instance of BaseRetriever.
        """

        pass

    @abstractmethod
    def embed(self) -> WrapperResponse:
        """
        Generate embeddings for the provided input using the specified Retriever.
        Only available if the Retriever is set up.

        :return: A dictionary containing the generated embeddings and model information.

        :raises ValueError: If the retriever is not set or if the client is not initialized.
        :raises EmbeddingError: If there is an error during the embedding process.
        """

        pass

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        is_rag: bool,
        max_tokens: int,
        **kwargs: Any
    ) -> WrapperResponse:
        """
        Generate a response from the LLM model based on the provided prompt.
        
        :param system_prompt: The system prompt to set the context for the model.
        :param user_prompt: The user prompt to generate a response for.
        :param max_tokens: The maximum number of tokens to generate.
        :param kwargs: Additional parameters for the OpenAI API.
        :return: A dictionary containing the generated response.

        :raises ValueError: If the model is not initialized or if the retriever is not set.
        :raises EmbeddingError: If there is an error during the generation process.
        """

        pass
