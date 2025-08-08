import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import main

def test_smoke():
    """
    A simple smoke test that runs the main simulation.
    """
    try:
        main()
    except Exception as e:
        assert False, f"Smoke test failed with exception: {e}"
