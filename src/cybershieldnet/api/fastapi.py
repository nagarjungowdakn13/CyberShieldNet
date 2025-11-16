from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from typing import Dict, List, Optional
import logging
import time
from contextlib import asynccontextmanager

from .schemas import (
    ThreatPredictionRequest, 
    ThreatPredictionResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfo
)
from .endpoints import router

logger = logging.getLogger(__name__)

class CyberShieldNetAPI:
    """CyberShieldNet FastAPI application"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = self._create_app()
        self._setup_middleware()
        self._setup_routes()
        self.model = None
        self.metrics = {}
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
        
        return FastAPI(
            title="CyberShieldNet API",
            description="Multi-Modal Fusion Framework for Predictive Threat Intelligence",
            version="1.0.0",
            lifespan=lifespan
        )
    
    def _setup_middleware(self):
        """Setup application middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors', {}).get('origins', ["*"]),
            allow_credentials=True,
            allow_methods=self.config.get('cors', {}).get('methods', ["*"]),
            allow_headers=self.config.get('cors', {}).get('headers', ["*"]),
        )
        
        # Trusted hosts middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.get('trusted_hosts', ["*"])
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        self.app.include_router(router, prefix="/api/v1")
        
        # Root endpoint
        @self.app.get("/")
        async def root():
            return {
                "message": "CyberShieldNet API",
                "version": "1.0.0",
                "status": "operational"
            }
    
    async def startup(self):
        """Application startup tasks"""
        logger.info("Starting CyberShieldNet API...")
        
        try:
            # Initialize model
            from ...core.base_model import CyberShieldNet
            self.model = CyberShieldNet()
            
            # Load pre-trained weights if available
            model_path = self.config.get('model_path')
            if model_path:
                self.model.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning("No pre-trained model loaded - using initialized weights")
            
            # Initialize metrics
            self.metrics = {
                'startup_time': time.time(),
                'requests_processed': 0,
                'successful_predictions': 0,
                'failed_predictions': 0
            }
            
            logger.info("CyberShieldNet API started successfully")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Application shutdown tasks"""
        logger.info("Shutting down CyberShieldNet API...")
        
        # Cleanup resources
        if self.model:
            del self.model
        
        logger.info("CyberShieldNet API shutdown complete")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs
        )

# Global API instance
api_instance: Optional[CyberShieldNetAPI] = None

def create_api(config: Dict) -> CyberShieldNetAPI:
    """Create and configure CyberShieldNet API"""
    global api_instance
    api_instance = CyberShieldNetAPI(config)
    return api_instance

def get_api() -> CyberShieldNetAPI:
    """Get the global API instance"""
    if api_instance is None:
        raise RuntimeError("API not initialized. Call create_api first.")
    return api_instance

app = FastAPI(
    title="CyberShieldNet API",
    description="Multi-Modal Fusion Framework for Predictive Threat Intelligence",
    version="1.0.0"
)

# Include router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "CyberShieldNet API",
        "version": "1.0.0",
        "status": "operational"
    }

if __name__ == "__main__":
    # For development
    uvicorn.run("src.cybershieldnet.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)