"""
conftest.py
-----------
Shared pytest configuration and fixtures.
Ensures the package root is on sys.path so tests can be run
from any directory with:  pytest tests/ -v
"""
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
