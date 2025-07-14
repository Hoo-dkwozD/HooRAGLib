#!/usr/bin/env python3

"""
Test suite for OpenAI LLM in HooRAGLib.

:author: Hoo-dkwozD
:version: 1.0.0
:date: 2025-07-14
"""

# Python Standard Library imports

# Third-party imports
import dotenv
import pytest

# Local imports
import HooRAGLib.Models.OpenAI
from HooRAGLib.Retrievers.BaseRetriever import BaseRetriever

@pytest.fixture
def mock_retriever(mocker):
    """Fixture to mock the retriever object."""

    return mocker.Mock()

@pytest.fixture
def mock_openai_client(mocker):
    """Fixture to mock the OpenAI client."""

    class MockOpenAIClient:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.models = mocker.Mock()
            self.chat = mocker.Mock()

            class MockModel:
                def __init__(self, id):
                    self.id = id
            mock_models = ["gpt-3.5-turbo", "gpt-4"]
            self.models.list.data = [MockModel(id=model) for model in mock_models]

            class MockChatCompletionChoice:
                def __init__(self, message):
                    self.message = mocker.Mock()
                    self.message.content = message
            mock_messages = 'abcde'.split('')
            self.chat.completions.create.return_value = [MockChatCompletionChoice(message=msg) for msg in mock_messages]

    return mocker.patch('HooRAGLib.Models.OpenAI.OpenAI', return_value=MockOpenAIClient)

def test_openai_llm_initialization(mocker, mock_openai_client):
    """Test the initialization of the OpenAI LLM."""

    from HooRAGLib.Models.OpenAI import OpenAILLM
    model_name = "test-model"
    llm = OpenAILLM(model_name=model_name, test_1="test", test_2="test")
    assert llm.model_name == model_name
    assert llm.models == ["gpt-3.5-turbo", "gpt-4"]

    assert mock_openai_client.called
    assert mock_openai_client.call_args[1]['api_key'] == 'test-key'
    assert mock_openai_client.call_args[1]['test_1'] == 'test'
    assert mock_openai_client.call_args[1]['test_2'] == 'test'

def test_openai_llm_initialization_without_api_key(mocker, mock_openai_client):
    """Test initialization without an API key raises ValueError."""

    from HooRAGLib.Models.OpenAI import OpenAILLM

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set."):
        OpenAILLM(model_name="test-model", api_key=None)

def test_openai_llm_configure(mocker, mock_retriever, mock_openai_client):
    """Test the configuration of the OpenAI LLM."""

    mock_retriever.__class__ = BaseRetriever  # Ensure mock_retriever is a BaseRetriever instance

    from HooRAGLib.Models.OpenAI import OpenAILLM

    model_name = "test-model"
    llm = OpenAILLM(model_name=model_name, api_key="test-key")
    
    model_version = "gpt-3.5-turbo"
    system_prompt = "This is a test system prompt."

    response = llm.configure(model_version=model_version, system_prompt=system_prompt, retriever=mock_retriever)

    assert llm.model_version == model_version
    assert llm.system_prompt == system_prompt
    assert llm.retriever == mock_retriever

    assert response['status'] == True
    assert response['message'] == f"OpenAI model '{model_name}' configured successfully."
    assert response['model_version'] == model_version
    assert response['retriever'] == mock_retriever.__class__.__name__
