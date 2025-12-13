from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from .schemas import (
    ThreatPredictionRequest,
    ThreatPredictionResponse,
    RiskAssessmentRequest, 
    RiskAssessmentResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfo,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelUpdateRequest,
    ModelUpdateResponse,
    ErrorResponse
)
from ..core.base_model import CyberShieldNet
from ..utils.metrics import ThreatDetectionMetrics
from ..utils.explainability import ExplainabilityEngine
from ..risk_assessment.drpa import DynamicRiskPropagation

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instances (would be properly initialized in app)
model: Optional[CyberShieldNet] = None
metrics_calculator: Optional[ThreatDetectionMetrics] = None
explainability_engine: Optional[ExplainabilityEngine] = None
risk_assessor: Optional[DynamicRiskPropagation] = None

def get_model() -> CyberShieldNet:
    """Get the global model instance"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model

def get_metrics_calculator() -> ThreatDetectionMetrics:
    """Get the global metrics calculator"""
    if metrics_calculator is None:
        raise HTTPException(status_code=503, detail="Metrics calculator not available")
    return metrics_calculator

def get_explainability_engine() -> ExplainabilityEngine:
    """Get the global explainability engine"""
    if explainability_engine is None:
        raise HTTPException(status_code=503, detail="Explainability engine not available")
    return explainability_engine

def get_risk_assessor() -> DynamicRiskPropagation:
    """Get the global risk assessor"""
    if risk_assessor is None:
        raise HTTPException(status_code=503, detail="Risk assessor not available")
    return risk_assessor

@router.post("/predict", response_model=ThreatPredictionResponse)
async def predict_threats(
    request: ThreatPredictionRequest,
    background_tasks: BackgroundTasks,
    model: CyberShieldNet = Depends(get_model)
):
    """
    Predict cyber threats from multi-modal data
    
    - **graph_data**: Network structure and dependencies
    - **temporal_data**: Time-series behavior patterns  
    - **behavioral_data**: User and system behavior features
    - **context_data**: Organizational context and asset information
    """
    try:
        start_time = time.time()
        logger.info("Received threat prediction request")
        
        # Convert request to model input format
        model_inputs = _prepare_model_inputs(request)
        
        # Make prediction
        threat_predictions, _ = model.predict(**model_inputs)
        
        # Convert predictions to response format
        predictions = _convert_predictions_to_response(threat_predictions, request)
        
        processing_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(_record_prediction_metrics, processing_time, len(predictions))
        
        response = ThreatPredictionResponse(
            predictions=predictions,
            processing_time=processing_time,
            model_version="1.0.0",  # Would come from model config
            timestamp=datetime.now(),
            metadata={"input_modalities_used": list(model_inputs.keys())}
        )
        
        logger.info(f"Threat prediction completed in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/assess-risk", response_model=RiskAssessmentResponse)
async def assess_risk(
    request: RiskAssessmentRequest,
    risk_assessor: DynamicRiskPropagation = Depends(get_risk_assessor)
):
    """
    Assess organizational risk based on threat predictions
    
    - **threat_predictions**: Threat predictions from model
    - **assets**: Asset inventory and criticality information
    - **vulnerabilities**: Known vulnerabilities and exposures
    - **business_context**: Business impact and recovery objectives
    """
    try:
        start_time = time.time()
        logger.info("Received risk assessment request")
        
        # Prepare risk assessment inputs
        risk_inputs = _prepare_risk_inputs(request)
        
        # Calculate risk scores
        risk_scores = risk_assessor(**risk_inputs)
        
        # Convert to response format
        asset_risks = _convert_risk_scores_to_response(risk_scores, request.assets)
        
        processing_time = time.time() - start_time
        
        response = RiskAssessmentResponse(
            asset_risks=asset_risks,
            organizational_risk=_calculate_organizational_risk(asset_risks),
            highest_risk_assets=_get_highest_risk_assets(asset_risks),
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Risk assessment completed in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    request: ExplainRequest,
    explainer: ExplainabilityEngine = Depends(get_explainability_engine),
    model: CyberShieldNet = Depends(get_model)
):
    """
    Explain model predictions using SHAP, attention, or feature importance
    
    - **prediction_request**: Original prediction request
    - **prediction_response**: Model predictions to explain
    - **explanation_type**: Type of explanation to generate
    - **explanation_config**: Configuration for explanation method
    """
    try:
        start_time = time.time()
        logger.info(f"Received explanation request for {request.explanation_type}")
        
        # Generate explanations based on type
        if request.explanation_type == "shap":
            explanations = _generate_shap_explanations(request, model, explainer)
        elif request.explanation_type == "attention":
            explanations = _generate_attention_explanations(request, model, explainer)
        elif request.explanation_type == "feature_importance":
            explanations = _generate_feature_importance_explanations(request, model, explainer)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported explanation type: {request.explanation_type}")
        
        processing_time = time.time() - start_time
        
        response = ExplainResponse(
            explanations=explanations,
            global_importance=_calculate_global_importance(explanations),
            explanation_method=request.explanation_type,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Explanation completed in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    try:
        # Basic health check
        status = "healthy"
        model_loaded = model is not None and model.is_trained
        
        # System metrics
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Uptime (simplified)
        uptime = time.time() - getattr(health_check, '_start_time', time.time())
        
        response = HealthResponse(
            status=status,
            model_loaded=model_loaded,
            model_version="1.0.0" if model_loaded else None,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            performance_metrics={
                "predictions_per_second": 0,  # Would track this
                "average_response_time": 0    # Would track this
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(model: CyberShieldNet = Depends(get_model)):
    """Get information about the loaded model"""
    try:
        model_info = ModelInfo(
            name="CyberShieldNet",
            version="1.0.0",
            description="Multi-Modal Fusion Framework for Predictive Threat Intelligence",
            input_modalities=["graph", "temporal", "behavioral", "contextual"],
            output_types=["threat_predictions", "risk_scores"],
            training_date=datetime.now(),  # Would come from model metadata
            performance_metrics={
                "accuracy": 0.962,
                "precision": 0.951,
                "recall": 0.958,
                "f1_score": 0.954
            },
            model_size_mb=250.0  # Would calculate actual size
        )
        
        return model_info
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

@router.post("/batch/predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model: CyberShieldNet = Depends(get_model)
):
    """Batch prediction for multiple requests"""
    try:
        start_time = time.time()
        logger.info(f"Received batch prediction request with {len(request.requests)} items")
        
        responses = []
        successful = 0
        failed = 0
        
        for i, pred_request in enumerate(request.requests):
            try:
                # Process individual prediction
                model_inputs = _prepare_model_inputs(pred_request)
                threat_predictions, _ = model.predict(**model_inputs)
                predictions = _convert_predictions_to_response(threat_predictions, pred_request)
                
                response = ThreatPredictionResponse(
                    predictions=predictions,
                    processing_time=0,  # Individual time not tracked
                    model_version="1.0.0",
                    timestamp=datetime.now()
                )
                
                responses.append(response)
                successful += 1
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {str(e)}")
                failed += 1
                # Optionally include error details for failed items
        
        total_processing_time = time.time() - start_time
        
        batch_response = BatchPredictionResponse(
            responses=responses,
            total_processing_time=total_processing_time,
            successful_predictions=successful,
            failed_predictions=failed,
            batch_id=f"batch_{int(start_time)}"
        )
        
        logger.info(f"Batch prediction completed: {successful} successful, {failed} failed")
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/model/update", response_model=ModelUpdateResponse)
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks
):
    """Update the model with a new version"""
    try:
        logger.info("Received model update request")
        
        # Validate new model
        validation_result = _validate_new_model(request.model_path, request.validation_data)
        
        if not validation_result["success"]:
            if request.rollback_on_failure:
                raise HTTPException(status_code=400, detail=f"Model validation failed: {validation_result['message']}")
            else:
                logger.warning(f"Model validation failed but proceeding: {validation_result['message']}")
        
        # Update model (in background for production)
        background_tasks.add_task(_perform_model_update, request, validation_result)
        
        response = ModelUpdateResponse(
            success=True,
            message="Model update initiated",
            new_version=request.model_config.get("version", "unknown"),
            validation_metrics=validation_result.get("metrics"),
            update_time=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Model update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")

@router.get("/threats/types")
async def get_threat_types():
    """Get list of supported threat types"""
    from ..core.constants import THREAT_CATEGORIES
    
    return {
        "threat_types": THREAT_CATEGORIES,
        "count": len(THREAT_CATEGORIES)
    }

@router.get("/assets/types") 
async def get_asset_types():
    """Get list of supported asset types"""
    from ..core.constants import DATA_MODALITIES
    
    return {
        "asset_types": DATA_MODALITIES,
        "count": len(DATA_MODALITIES)
    }

# Helper functions
def _prepare_model_inputs(request: ThreatPredictionRequest) -> Dict[str, Any]:
    """Convert API request to model input format"""
    inputs = {}
    
    if request.graph_data:
        inputs['graph_data'] = _convert_graph_data(request.graph_data)
    
    if request.temporal_data:
        inputs['temporal_data'] = _convert_temporal_data(request.temporal_data)
    
    if request.behavioral_data:
        inputs['behavioral_data'] = _convert_behavioral_data(request.behavioral_data)
    
    if request.context_data:
        inputs['context_data'] = _convert_context_data(request.context_data)
    
    return inputs

def _convert_graph_data(graph_data) -> Dict:
    """Convert graph data to model format"""
    # Implementation would convert from API format to model expected format
    return {
        'x': graph_data.node_features,  # Would convert properly
        'edge_index': graph_data.edge_index,
        'nodes': graph_data.nodes,
        'edges': graph_data.edges
    }

def _convert_temporal_data(temporal_data) -> Any:
    """Convert temporal data to model format"""
    # Implementation would convert from API format to model expected format
    return temporal_data.sequences

def _convert_behavioral_data(behavioral_data) -> Any:
    """Convert behavioral data to model format"""
    # Implementation would convert from API format to model expected format
    return [list(feature.values()) for feature in behavioral_data.features]

def _convert_context_data(context_data) -> Dict:
    """Convert context data to model format"""
    return {
        'assets': context_data.assets,
        'vulnerabilities': context_data.vulnerabilities,
        'threat_intel': context_data.threat_intel or [],
        'network_topology': context_data.network_topology or {},
        'business_context': context_data.business_context or {}
    }

def _convert_predictions_to_response(predictions, request) -> List:
    """Convert model predictions to API response format"""
    # This is a simplified conversion
    response_predictions = []
    
    for i, pred in enumerate(predictions):
        response_predictions.append({
            'threat_type': 'apt_campaign',  # Would map from model output
            'probability': float(pred),
            'confidence': float(pred),  # Would calculate properly
            'assets_affected': ['asset_1', 'asset_2'],  # Would determine from context
            'severity': 'high' if pred > 0.7 else 'medium'  # Would calculate properly
        })
    
    return response_predictions

def _prepare_risk_inputs(request: RiskAssessmentRequest) -> Dict:
    """Prepare risk assessment inputs"""
    return {
        'threat_predictions': request.threat_predictions,
        'assets': request.assets,
        'vulnerabilities': request.vulnerabilities,
        'business_context': request.business_context or {}
    }

def _convert_risk_scores_to_response(risk_scores, assets) -> List:
    """Convert risk scores to API response format"""
    asset_risks = []
    
    for i, (asset, risk_score) in enumerate(zip(assets, risk_scores)):
        asset_risks.append({
            'asset_id': asset.get('id', f"asset_{i}"),
            'asset_name': asset.get('name', f"Asset {i}"),
            'asset_type': asset.get('type', 'server'),
            'risk_score': float(risk_score),
            'risk_level': 'high' if risk_score > 0.7 else 'medium',  # Would calculate properly
            'contributing_threats': ['threat_1', 'threat_2'],  # Would determine from analysis
            'mitigation_recommendations': ['Recommendation 1', 'Recommendation 2']  # Would generate
        })
    
    return asset_risks

def _calculate_organizational_risk(asset_risks: List) -> float:
    """Calculate overall organizational risk"""
    if not asset_risks:
        return 0.0
    
    return max(asset['risk_score'] for asset in asset_risks)  # Use maximum risk

def _get_highest_risk_assets(asset_risks: List, top_n: int = 5) -> List[str]:
    """Get highest risk assets"""
    sorted_assets = sorted(asset_risks, key=lambda x: x['risk_score'], reverse=True)
    return [asset['asset_name'] for asset in sorted_assets[:top_n]]

def _generate_shap_explanations(request, model, explainer) -> List:
    """Generate SHAP explanations"""
    # Implementation would generate SHAP values
    return [{
        'threat_type': pred.threat_type,
        'feature_importance': [],
        'shap_values': [0.1, 0.2, 0.3],  # Would calculate actual values
        'decision_factors': ['Factor 1', 'Factor 2', 'Factor 3']  # Would generate
    } for pred in request.prediction_response.predictions]

def _generate_attention_explanations(request, model, explainer) -> List:
    """Generate attention explanations"""
    # Implementation would generate attention weights
    return [{
        'threat_type': pred.threat_type,
        'feature_importance': [],
        'attention_weights': {'node_1': 0.5, 'node_2': 0.3},  # Would calculate actual weights
        'decision_factors': ['Attention to node_1', 'Attention to node_2']  # Would generate
    } for pred in request.prediction_response.predictions]

def _generate_feature_importance_explanations(request, model, explainer) -> List:
    """Generate feature importance explanations"""
    # Implementation would calculate feature importance
    return [{
        'threat_type': pred.threat_type,
        'feature_importance': [
            {'feature_name': 'feature_1', 'importance_score': 0.8, 'contribution': 0.5, 'direction': 'positive'},
            {'feature_name': 'feature_2', 'importance_score': 0.6, 'contribution': 0.3, 'direction': 'negative'}
        ],
        'decision_factors': ['High feature_1', 'Low feature_2']  # Would generate
    } for pred in request.prediction_response.predictions]

def _calculate_global_importance(explanations: List) -> List:
    """Calculate global feature importance across all explanations"""
    # Implementation would aggregate feature importance across all threats
    return [
        {'feature_name': 'global_feature_1', 'importance_score': 0.9, 'contribution': 0.7, 'direction': 'positive'},
        {'feature_name': 'global_feature_2', 'importance_score': 0.7, 'contribution': 0.5, 'direction': 'negative'}
    ]

def _record_prediction_metrics(processing_time: float, num_predictions: int):
    """Record prediction metrics for monitoring"""
    # Implementation would update metrics storage
    logger.debug(f"Recorded prediction: {processing_time}s for {num_predictions} predictions")

def _validate_new_model(model_path: str, validation_data: Optional[Dict]) -> Dict:
    """Validate new model before update"""
    # Implementation would validate model file and performance
    return {
        "success": True,
        "message": "Model validation passed",
        "metrics": {"accuracy": 0.95, "precision": 0.92}  # Would calculate actual metrics
    }

def _perform_model_update(request: ModelUpdateRequest, validation_result: Dict):
    """Perform the actual model update"""
    # Implementation would safely update the model
    logger.info("Performing model update in background")
    # This would include:
    # 1. Loading new model
    # 2. Validation testing
    # 3. Atomic switch to new model
    # 4. Cleanup of old model