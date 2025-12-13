from pydantic import BaseModel, Field, validator
try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = object
    def Field(default=None, **kwargs):
        return default
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class ThreatType(str, Enum):
    APT_CAMPAIGN = "apt_campaign"
    RANSOMWARE = "ransomware"
    INSIDER_THREAT = "insider_threat"
    DDOS_ATTACK = "ddos_attack"
    ZERO_DAY_EXPLOIT = "zero_day_exploit"
    PHISHING_CAMPAIGN = "phishing_campaign"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CREDENTIAL_THEFT = "credential_theft"

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class AssetType(str, Enum):
    SERVER = "server"
    WORKSTATION = "workstation"
    NETWORK_DEVICE = "network_device"
    APPLICATION = "application"
    DATABASE = "database"
    CLOUD_SERVICE = "cloud_service"
    IOT_DEVICE = "iot_device"

class GraphData(BaseModel):
    """Graph structure data for threat analysis"""
    nodes: List[Dict[str, Any]] = Field(..., description="List of nodes with features")
    edges: List[Dict[str, Any]] = Field(..., description="List of edges with attributes")
    node_features: Optional[List[List[float]]] = Field(None, description="Node feature matrix")
    edge_index: Optional[List[List[int]]] = Field(None, description="Edge index for graph convolution")

class TemporalData(BaseModel):
    """Temporal sequence data for threat analysis"""
    sequences: List[List[float]] = Field(..., description="Temporal sequences")
    timestamps: Optional[List[datetime]] = Field(None, description="Sequence timestamps")
    sequence_length: int = Field(..., description="Length of each sequence")
    feature_dim: int = Field(..., description="Dimension of temporal features")

class BehavioralData(BaseModel):
    """Behavioral feature data for threat analysis"""
    features: List[Dict[str, float]] = Field(..., description="Behavioral features")
    feature_names: List[str] = Field(..., description="Names of behavioral features")
    timestamps: Optional[List[datetime]] = Field(None, description="Feature timestamps")

class ContextData(BaseModel):
    """Contextual information for risk assessment"""
    assets: List[Dict[str, Any]] = Field(..., description="Asset information")
    vulnerabilities: List[Dict[str, Any]] = Field(..., description="Vulnerability data")
    threat_intel: Optional[List[Dict[str, Any]]] = Field(None, description="Threat intelligence feeds")
    network_topology: Optional[Dict[str, Any]] = Field(None, description="Network topology information")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Business context data")

class ThreatPredictionRequest(BaseModel):
    """Request model for threat prediction"""
    graph_data: Optional[GraphData] = Field(None, description="Graph structure data")
    temporal_data: Optional[TemporalData] = Field(None, description="Temporal sequence data")
    behavioral_data: Optional[BehavioralData] = Field(None, description="Behavioral feature data")
    context_data: Optional[ContextData] = Field(None, description="Contextual information")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration overrides")

class ThreatPrediction(BaseModel):
    """Individual threat prediction result"""
    threat_type: ThreatType = Field(..., description="Type of detected threat")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    assets_affected: List[str] = Field(..., description="List of affected assets")
    severity: RiskLevel = Field(..., description="Threat severity level")
    
    # Severity should be provided; classification logic can be applied in service layer.

class ThreatPredictionResponse(BaseModel):
    """Response model for threat prediction"""
    predictions: List[ThreatPrediction] = Field(..., description="List of threat predictions")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ThreatPredictionResponse(BaseModel):
    """Response model for threat prediction"""
    predictions: List[ThreatPrediction] = Field(..., description="List of threat predictions")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class AssetRisk(BaseModel):
    """Individual asset risk assessment"""
    asset_id: str = Field(..., description="Unique asset identifier")
    asset_name: str = Field(..., description="Asset name")
    asset_type: AssetType = Field(..., description="Type of asset")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calculated risk score")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    contributing_threats: List[str] = Field(..., description="Threats contributing to risk")
    mitigation_recommendations: List[str] = Field(..., description="Recommended mitigation actions")
    
    # Risk level to be provided or computed outside of schema.

class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment"""
    asset_risks: List[AssetRisk] = Field(..., description="Risk assessments for all assets")
    organizational_risk: float = Field(..., ge=0.0, le=1.0, description="Overall organizational risk score")
    highest_risk_assets: List[str] = Field(..., description="List of highest risk assets")
    risk_propagation_paths: Optional[Dict[str, List[str]]] = Field(None, description="Risk propagation paths")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Assessment timestamp")

class ExplainRequest(BaseModel):
    """Request model for model explanation"""
    prediction_request: ThreatPredictionRequest = Field(..., description="Original prediction request")
    prediction_response: ThreatPredictionResponse = Field(..., description="Prediction response to explain")
    explanation_type: str = Field(..., description="Type of explanation (shap, attention, feature_importance)")
    explanation_config: Optional[Dict[str, Any]] = Field(None, description="Explanation configuration")

class FeatureImportance(BaseModel):
    """Feature importance explanation"""
    feature_name: str = Field(..., description="Name of the feature")
    importance_score: float = Field(..., description="Importance score")
    contribution: float = Field(..., description="Contribution to prediction")
    direction: str = Field(..., description="Direction of effect (positive/negative)")

class ExplanationResult(BaseModel):
    """Individual explanation result"""
    threat_type: ThreatType = Field(..., description="Threat type being explained")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importance scores")
    attention_weights: Optional[Dict[str, float]] = Field(None, description="Attention weights if available")
    shap_values: Optional[List[float]] = Field(None, description="SHAP values if computed")
    decision_factors: List[str] = Field(..., description="Key decision factors in natural language")

class ExplainResponse(BaseModel):
    """Response model for model explanation"""
    explanations: List[ExplanationResult] = Field(..., description="Explanation results for each threat")
    global_importance: List[FeatureImportance] = Field(..., description="Global feature importance")
    explanation_method: str = Field(..., description="Explanation method used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Explanation timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    last_prediction_time: Optional[datetime] = Field(None, description="Time of last prediction")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")

class ModelInfo(BaseModel):
    """Model information response"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: str = Field(..., description="Model description")
    input_modalities: List[str] = Field(..., description="Supported input modalities")
    output_types: List[str] = Field(..., description="Supported output types")
    training_date: Optional[datetime] = Field(None, description="Date model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    model_size_mb: float = Field(..., description="Model size in MB")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[ThreatPredictionRequest] = Field(..., description="List of prediction requests")
    batch_config: Optional[Dict[str, Any]] = Field(None, description="Batch processing configuration")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    responses: List[ThreatPredictionResponse] = Field(..., description="List of prediction responses")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    batch_id: Optional[str] = Field(None, description="Batch identifier for tracking")

class ModelUpdateRequest(BaseModel):
    """Model update request"""
    model_path: str = Field(..., description="Path to new model file")
    model_config: Dict[str, Any] = Field(..., description="New model configuration")
    validation_data: Optional[Dict[str, Any]] = Field(None, description="Validation data for model testing")
    rollback_on_failure: bool = Field(True, description="Whether to rollback on update failure")

class ModelUpdateResponse(BaseModel):
    """Model update response"""
    success: bool = Field(..., description="Update success status")
    message: str = Field(..., description="Update message")
    new_version: Optional[str] = Field(None, description="New model version")
    validation_metrics: Optional[Dict[str, float]] = Field(None, description="Validation metrics")
    update_time: datetime = Field(..., description="Update timestamp")