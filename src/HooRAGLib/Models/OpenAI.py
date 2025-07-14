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
from typing import Any, Optional

# Third-party imports
from openai import OpenAI, AuthenticationError, APIError, RateLimitError

# Local imports
from HooRAGLib.Helpers.Errors import EmbeddingError
from HooRAGLib.Models.BaseLLM import BaseLLM
from HooRAGLib.Models.BaseLLM import WrapperResponse
from HooRAGLib.Retrievers.BaseRetriever import BaseRetriever


class OpenAILLM(BaseLLM):
    """
    Wrapper class that interacts with OpenAI API.
    Inherits from BaseModel.
    """

    def __init__(
        self, 
        model_name: str,
        **kwargs: Any
    ):
        """
        Initialize the OpenAI wrapper with a model name and optional parameters.
        
        :param model_name: The name of the OpenAI model to use.
        :param kwargs: Additional parameters for the OpenAI API.

        :raises ValueError: If the api key is not provided.
        :raises AuthenticationError: If the provided API key is invalid.
        """

        API_KEY = os.environ.get('OPENAI_API_KEY') if 'api_key' not in kwargs else kwargs.get('api_key')

        if not API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        try:
            client = OpenAI(api_key=API_KEY, **kwargs)
            models = client.models.list()
        except AuthenticationError as e:
            raise ValueError("Invalid OpenAI API key provided.") from e

        self.client = client
        self.models = models
        self.model_name = model_name

        self.model_version = None
        self.system_prompt = None
        self.retriever = None

    def configure(
        self, 
        model_version: str,
        system_prompt: Optional[str] = None,
        retriever: Optional[BaseRetriever] = None,
    ) -> WrapperResponse:
        """
        Configure the LLM Wrapper with necessary settings & components.
        
        :param model_version: The version of the model to use.
        :param retriever: An optional BaseRetriever instance for RAG.

        :return: A dictionary containing the configuration response.

        :raises ValueError: If the model version is not specified or invalid.
        :raises TypeError: If the retriever is not an instance of BaseRetriever.
        """

        # Check if model is instantiated
        self._check_client()

        # Check for valid model version
        if not model_version:
            raise ValueError("Model version must be specified.")
        # Check if model_version is a valid OpenAI model
        if model_version not in [model.id for model in self.models.data]:
            raise ValueError(f"Model version '{model_version}' is not available as an OpenAI models.")
        self.model_version = model_version

        self.system_prompt = system_prompt

        if retriever is not None and not isinstance(retriever, BaseRetriever):
            raise TypeError("Retriever must be an instance of BaseRetriever.")
        self.retriever = retriever

        return {
            "status": True,
            "message": f"OpenAI model '{self.model_name}' configured successfully.",
            "model_version": self.model_version,
            "retriever": self.retriever.__class__.__name__
        }

    def embed(self) -> WrapperResponse:
        """
        Generate embeddings for the provided input using the specified Retriever.
        Only available if the Retriever is set up.

        :return: A dictionary containing the generated embeddings and model information.

        :raises ValueError: If the retriever is not set or if the client is not initialized.
        :raises EmbeddingError: If there is an error during the embedding process.
        """

        # Check if model is instantiated
        self._check_client()

        # Check if retriever is set
        self._check_retriever()

        # Use the retriever to generate embeddings
        try:
            response = self.retriever.embed(client=self.client)

            return {
                "status": True,
                "message": "Embeddings generated successfully.",
                "embeddings": response['data'],
                "model": self.model_name
            }
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def generate(
        self,
        user_prompt: str,
        is_rag: bool = True,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> WrapperResponse:
        """
        Generate a response from the OpenAI model based on the provided prompt.
        
        :param system_prompt: The system prompt to set the context for the model.
        :param user_prompt: The user prompt to generate a response for.
        :param max_tokens: The maximum number of tokens to generate.
        :param kwargs: Additional parameters for the OpenAI API.
        :return: A dictionary containing the generated response.

        :raises ValueError: If the model is not initialized or if the retriever is not set.
        :raises EmbeddingError: If there is an error during the generation process.
        """

        # Check if model is instantiated
        self._check_client()

        if is_rag:
            # Ensure retriever is set for RAG
            self._check_retriever()

            # Check if retriever has embeddings generated
            self._check_embeddings()

            # Generate response using OpenAI's chat completions
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=[
                    {
                        "role": "system", 
                        "content": self.system_prompt if 'system_prompt' not in kwargs else kwargs.get('system_prompt', 'You are a helpful assistant.')
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    },
                    {
                        "role": "assistant", 
                        "content": self.retriever.retrieve(user_prompt)
                    }
                ],
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=[
                    {
                        "role": "system", 
                        "content": self.system_prompt if 'system_prompt' not in kwargs else kwargs.get('system_prompt', 'You are a helpful assistant.')
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                max_tokens=max_tokens,
                **kwargs
            )

        # Return all generated choices and metadata 
        return {
            "status": True,
            "message": "Response generated successfully.",
            "choices": [choice.message.content for choice in response.choices],
            "model": self.model_name,
            "usage": response.usage
        }

    def _check_client(self) -> None:
        """
        Check if the OpenAI client is initialized.
        
        :raises ValueError: If the OpenAI client is not initialized.
        """

        if not hasattr(self, 'client') and not hasattr(self, 'models'):
            raise ValueError("OpenAI client is not initialized.")

    def _check_retriever(self) -> None:
        """
        Check if the retriever is set.
        
        :raises ValueError: If the retriever is not set.
        """

        if not hasattr(self, 'retriever'):
            raise ValueError("Retriever is not set. Please configure the retriever before embedding.")

    def _check_embeddings(self) -> None:
        """
        Check if the embedding method is available.
        
        :raises ValueError: If the retriever is not set.
        :raises EmbeddingError: If the retriever does not have embeddings generated.
        """

        self._check_retriever()

        if not self.retriever.has_embedding():
            raise EmbeddingError("Embeddings have not been generated. Please call the embed() method first.")
