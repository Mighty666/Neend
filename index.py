"""
FastAPI entrypoint for deployment platforms.
This file imports the app from src.api.main to satisfy deployment requirements.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.main import app

__all__ = ["app"]

