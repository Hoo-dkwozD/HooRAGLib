#!/usr/bin/env python3

"""
Setup script for HooRAGLib tests.

:author: Hoo-dkwozD
:version: 1.0.0
:date: 2025-07-14
"""

# Python Standard Library imports

# Third-party imports
import dotenv

# Local imports

def pytest_configure(config):
    """Configure pytest settings."""

    print("[Pytest Setup] Setting up HooRAGLib test environment... ")

    # Add any necessary configuration here
    try: 
        dotenv.load_dotenv("./../.test.env")
    except:
        print("[Pytest Setup] Error: .test.env file not found.")

        raise FileNotFoundError(
            "The .test.env file is required for testing but was not found. Please create it in the root directory of the project."
        )

    print("[Pytest Setup] HooRAGLib test environment setup complete.")
