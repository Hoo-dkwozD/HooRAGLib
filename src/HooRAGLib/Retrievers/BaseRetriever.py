#!/usr/bin/env python3

"""
Base class for RAG Retriever Class in HooRAGLib.
This class defines the interface for retrievers that can be used with LLMs in HooRAGLib.

:author: Hoo-dkwozD
:version: 1.0.0
:date: 2025-07-14
"""

# Python Standard Library imports

# Third-party imports

# Local imports


class BaseRetriever:
    """
    Base class for all retrievers.
    This class should be inherited by all retriever implementations.
    It provides a common interface for retrieval operations.
    """

    def retrieve(self, query: str, top_k: int = 10):
        """
        Retrieve documents based on the query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of retrieved documents.
        """
        raise NotImplementedError("Subclasses must implement this method.")
