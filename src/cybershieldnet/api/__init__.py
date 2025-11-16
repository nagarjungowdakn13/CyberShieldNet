"""API modules for CyberShieldNet

Expose the FastAPI `app` and router for easy inclusion by external
servers or development utilities.
"""

from .endpoints import router
from .fastapi import app, create_api, CyberShieldNetAPI

__all__ = ['router', 'app', 'create_api', 'CyberShieldNetAPI']