"""
============================================
EVIDENTLY AI - DRIFT DETECTION SERVICE
============================================

This service provides:
1. Data drift detection
2. Model performance monitoring
3. Report generation
4. Prometheus metrics exposure
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from pathlib import Path

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns

# Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
REPORTS_DIR = Path("/app/reports")
DATA_DIR = Path("/app/data")
REFERENCE_DIR = Path("/app/reference")

# Create directories if they don't exist
REPORTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

# ============================================
# PROMETHEUS METRICS
# ============================================

# Drift metrics
DRIFT_DETECTED = Gauge(
    'evidently_data_drift_detected',
    'Whether data drift is detected (1=yes, 0=no)'
)

DRIFT_SCORE = Gauge(
    'evidently_drift_score',
    'Overall drift score',
)

FEATURE_DRIFT = Gauge(
    'evidently_feature_drift',
    'Drift detected per feature',
    ['feature_name']
)

DRIFTED_FEATURES_COUNT = Gauge(
    'evidently_drifted_features_count',
    'Number of features with detected drift'
)

# Analysis metrics
ANALYSIS_COUNT = Counter(
    'evidently_analysis_total',
    'Total number of drift analyses performed'
)

ANALYSIS_DURATION = Histogram(
    'evidently_analysis_duration_seconds',
    'Time taken to perform drift analysis',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Data quality metrics
MISSING_VALUES = Gauge(
    'evidently_missing_values_ratio',
    'Ratio of missing values',
    ['feature_name']
)

# ============================================
# PYDANTIC MODELS
# ============================================


FEATURE_NAMES = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

def validate_features(row):
    if set(row) != set(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail="Features must match the 13-column Wine training schema",
        )
    try:
        values = np.asarray(
            [row[name] for name in FEATURE_NAMES], dtype=float
        )
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Features must be numeric")
    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422, detail="Features must be finite numbers"
        )

class PredictionData(BaseModel):
    """Single prediction data point"""
    features: Dict[str, float]
    prediction: Optional[float] = None
    timestamp: Optional[str] = None
    model_version: Optional[str] = None

class BatchPredictionData(BaseModel):
    """Batch of prediction data"""
    data: List[Dict[str, Any]]
    feature_names: Optional[List[str]] = None

class DriftAnalysisRequest(BaseModel):
    """Request to trigger drift analysis"""
    window_size: int = Field(200, ge=100, le=10000)
    threshold: float = Field(0.3, gt=0, le=1, description="Share of drifted features required for dataset drift")

class ReferenceDataRequest(BaseModel):
    """Request to update reference data"""
    data: List[Dict[str, Any]]
    feature_names: List[str]
    description: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    reference_data_loaded: bool
    production_data_count: int
    last_analysis: Optional[str]
    reports_count: int

# ============================================
# DATA STORAGE (In-Memory for now)
# ============================================

class DataStore:
    """Simple in-memory data storage"""

    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.production_data: List[Dict] = []
        self.last_analysis_time: Optional[datetime] = None
        self.reference_metadata: Dict = {}

        # Load reference data if exists
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference data from disk if available"""
        reference_file = REFERENCE_DIR / "reference_data.csv"
        metadata_file = REFERENCE_DIR / "metadata.json"

        if reference_file.exists():
            try:
                self.reference_data = pd.read_csv(reference_file)
                logger.info(f"✅ Loaded reference data: {len(self.reference_data)} samples")

                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.reference_metadata = json.load(f)
                        logger.info(f"📋 Reference metadata: {self.reference_metadata.get('description', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ Failed to load reference data: {e}")

    def save_reference_data(self, data: pd.DataFrame, metadata: Dict = None):
        """Save reference data to disk"""
        try:
            reference_file = REFERENCE_DIR / "reference_data.csv"
            data.to_csv(reference_file, index=False)

            if metadata:
                metadata_file = REFERENCE_DIR / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            self.reference_data = data
            self.reference_metadata = metadata or {}
            logger.info(f"✅ Saved reference data: {len(data)} samples")
        except Exception as e:
            logger.error(f"❌ Failed to save reference data: {e}")
            raise

    def add_production_data(self, data: Dict):
        """Add production data point"""
        self.production_data.append(data)

        # Keep only last 10000 samples to avoid memory issues
        if len(self.production_data) > 10000:
            self.production_data = self.production_data[-10000:]

    def get_production_dataframe(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """Get production data as DataFrame"""
        if not self.production_data:
            return pd.DataFrame()

        data = self.production_data[-window_size:] if window_size else self.production_data
        return pd.DataFrame(data)

    def clear_production_data(self):
        """Clear production data"""
        self.production_data = []
        logger.info("🗑️ Cleared production data")

# Initialize data store
data_store = DataStore()

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Evidently AI - Drift Detection Service",
    description="Data drift detection and model monitoring service",
    version="1.0.0"
)

app_start_time = datetime.now()

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Evidently AI - Drift Detection",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "capture": "/capture (POST)",
            "analyze": "/analyze (POST)",
            "reports": "/reports",
            "reference": "/reference (GET/POST)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    reports = list(REPORTS_DIR.glob("*.html"))

    return HealthResponse(
        status="healthy",
        reference_data_loaded=data_store.reference_data is not None,
        production_data_count=len(data_store.production_data),
        last_analysis=data_store.last_analysis_time.isoformat() if data_store.last_analysis_time else None,
        reports_count=len(reports)
    )

@app.post("/capture")
async def capture_prediction(data: PredictionData):
    """Capture a single prediction for drift monitoring"""
    validate_features(data.features)
    try:
        # Add timestamp if not provided
        if not data.timestamp:
            data.timestamp = datetime.now().isoformat()

        # Convert to dict and store
        data_dict = {
            **data.features,
            'prediction': data.prediction,
            'timestamp': data.timestamp,
            'model_version': data.model_version
        }

        data_store.add_production_data(data_dict)

        return {
            "status": "success",
            "message": "Data captured successfully",
            "total_samples": len(data_store.production_data)
        }

    except Exception as e:
        logger.error(f"Error capturing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/capture/batch")
async def capture_batch(data: BatchPredictionData):
    """Capture batch of predictions"""
    # Validate the whole batch before storing any row.
    for item in data.data:
        validate_features({
            key: value for key, value in item.items()
            if key not in ("prediction", "timestamp", "model_version")
        })
    try:
        for item in data.data:
            data_store.add_production_data(item)

        return {
            "status": "success",
            "message": f"Captured {len(data.data)} samples",
            "total_samples": len(data_store.production_data)
        }

    except Exception as e:
        logger.error(f"Error capturing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_drift(
    request: DriftAnalysisRequest = DriftAnalysisRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger drift analysis"""
    try:
        # Check if reference data is available
        if data_store.reference_data is None:
            raise HTTPException(
                status_code=400,
                detail="Reference data not loaded. Please upload reference data first."
            )

        # Get production data
        production_df = data_store.get_production_dataframe(request.window_size)

        if len(production_df) < 100:
            raise HTTPException(
                status_code=400,
                detail="At least 100 production samples are required"
            )

        logger.info(f"🔍 Starting drift analysis...")
        logger.info(f"   Reference samples: {len(data_store.reference_data)}")
        logger.info(f"   Production samples: {len(production_df)}")

        # Perform drift analysis
        import time
        start_time = time.time()

        result = perform_drift_analysis(
            reference_data=data_store.reference_data,
            current_data=production_df,
            threshold=request.threshold
        )

        duration = time.time() - start_time

        # Update metrics
        ANALYSIS_COUNT.inc()
        ANALYSIS_DURATION.observe(duration)
        data_store.last_analysis_time = datetime.now()

        logger.info(f"✅ Analysis completed in {duration:.2f}s")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during drift analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reference")
async def get_reference_info():
    """Get reference data information"""
    if data_store.reference_data is None:
        return {
            "loaded": False,
            "message": "No reference data loaded"
        }

    return {
        "loaded": True,
        "samples": len(data_store.reference_data),
        "features": list(data_store.reference_data.columns),
        "metadata": data_store.reference_metadata
    }

@app.post("/reference")
async def upload_reference_data(request: ReferenceDataRequest):
    """Upload reference data"""
    try:
        # Convert to DataFrame
        if request.feature_names != FEATURE_NAMES:
            raise HTTPException(
                status_code=422,
                detail="Reference feature names or order are incorrect",
            )
        if len(request.data) < 100:
            raise HTTPException(
                status_code=422,
                detail="Reference requires at least 100 samples",
            )
        for row in request.data:
            validate_features(row)
        df = pd.DataFrame(request.data, columns=FEATURE_NAMES)

        # Validate
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Empty dataset provided")

        # Save reference data
        metadata = {
            "description": request.description or "Reference dataset",
            "uploaded_at": datetime.now().isoformat(),
            "samples": len(df),
            "features": request.feature_names or list(df.columns)
        }

        data_store.save_reference_data(df, metadata)

        return {
            "status": "success",
            "message": "Reference data uploaded successfully",
            "samples": len(df),
            "features": list(df.columns)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading reference data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports")
async def list_reports():
    """List all available reports"""
    reports = sorted(REPORTS_DIR.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True)

    return {
        "count": len(reports),
        "reports": [
            {
                "filename": r.name,
                "created": datetime.fromtimestamp(r.stat().st_mtime).isoformat(),
                "size_kb": round(r.stat().st_size / 1024, 2),
                "url": f"/reports/{r.name}"
            }
            for r in reports
        ]
    }

@app.get("/reports/{report_name}", response_class=HTMLResponse)
async def get_report(report_name: str):
    """Get specific report"""
    report_path = REPORTS_DIR / report_name

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return report_path.read_text()

@app.delete("/production-data")
async def clear_production_data():
    """Clear production data"""
    data_store.clear_production_data()
    return {"status": "success", "message": "Production data cleared"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================
# DRIFT ANALYSIS LOGIC
# ============================================

def perform_drift_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Drift score is the share of drifted features, not PSI."""
    for frame in (reference_data, current_data):
        if not set(FEATURE_NAMES).issubset(frame.columns):
            raise ValueError("Missing monitored Wine features")

    ref_df = reference_data[FEATURE_NAMES].astype(float).copy()
    curr_df = current_data[FEATURE_NAMES].astype(float).copy()
    if (
        not np.isfinite(ref_df.to_numpy()).all()
        or not np.isfinite(curr_df.to_numpy()).all()
    ):
        raise ValueError("Non-finite data cannot be analyzed")

    report = Report(metrics=[
        DataDriftPreset(drift_share=threshold),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_df, current_data=curr_df)
    results = report.as_dict()

    # DatasetDriftMetric contains the aggregate result.
    # Per-feature results are exported in a separate drift table.
    per_feature = {}
    for metric in results.get("metrics", []):
        result = metric.get("result", {})
        columns = result.get("drift_by_columns")
        if isinstance(columns, dict):
            per_feature.update(columns)

    if not set(FEATURE_NAMES).issubset(per_feature):
        raise RuntimeError(
            "Evidently report did not provide all 13 feature drift results"
        )

    drifted_features = []
    drift_scores = {}
    FEATURE_DRIFT.clear()
    MISSING_VALUES.clear()

    for feature in FEATURE_NAMES:
        info = per_feature[feature]
        detected = bool(info["drift_detected"])
        drift_scores[feature] = float(info["drift_score"])
        if detected:
            drifted_features.append(feature)
        FEATURE_DRIFT.labels(feature_name=feature).set(int(detected))
        MISSING_VALUES.labels(feature_name=feature).set(
            float(curr_df[feature].isna().mean())
        )

    drift_score = len(drifted_features) / len(FEATURE_NAMES)
    drift_detected = drift_score >= threshold
    DRIFT_DETECTED.set(int(drift_detected))
    DRIFT_SCORE.set(drift_score)
    DRIFTED_FEATURES_COUNT.set(len(drifted_features))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"drift_report_{timestamp}.html"
    report.save_html(str(REPORTS_DIR / filename))

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "score_definition": "share_of_drifted_features",
        "threshold": threshold,
        "drifted_features": drifted_features,
        "drift_scores": drift_scores,
        "total_features": len(FEATURE_NAMES),
        "drifted_count": len(drifted_features),
        "report_url": f"/reports/{filename}",
        "report_filename": filename,
        "reference_samples": len(ref_df),
        "current_samples": len(curr_df),
    }

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("="*50)
    logger.info("🚀 Starting Evidently AI Service")
    logger.info("="*50)

    if data_store.reference_data is not None:
        logger.info(f"✅ Reference data loaded: {len(data_store.reference_data)} samples")
        logger.info(f"   Features: {list(data_store.reference_data.columns)}")
    else:
        logger.warning("⚠️  No reference data loaded. Please upload reference data.")

    logger.info(f"📁 Reports directory: {REPORTS_DIR}")
    logger.info(f"📁 Data directory: {DATA_DIR}")
    logger.info(f"📁 Reference directory: {REFERENCE_DIR}")
    logger.info("="*50)
    logger.info("✅ Service ready!")
    logger.info("="*50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
