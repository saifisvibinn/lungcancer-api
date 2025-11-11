"""
FastAPI Lung Cancer Prediction API
A RESTful API for predicting lung cancer risk based on patient symptoms and characteristics.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import sys
import warnings
import os
import uvicorn

warnings.filterwarnings('ignore')

# Initialize FastAPI application
app = FastAPI(
    title="Lung Cancer Prediction API",
    description="A RESTful API for predicting lung cancer risk based on patient symptoms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model Loading with Compatibility Handling
# ============================================================================

model = None
scaler = None

# Try to load using the robust loader
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
    
    # First, try aggressive patching - USE EuclideanDistance64 (not 32!)
    try:
        import sklearn.metrics._dist_metrics as dist_metrics
        
        # Patch EuclideanDistance if missing - prioritize 64-bit version
        if not hasattr(dist_metrics, 'EuclideanDistance'):
            print("Attempting to patch EuclideanDistance...")
            
            # Try option 1: Use EuclideanDistance64 (model uses 64-bit)
            if hasattr(dist_metrics, 'EuclideanDistance64'):
                EuclideanDistance64 = dist_metrics.EuclideanDistance64
                dist_metrics.EuclideanDistance = EuclideanDistance64
                setattr(dist_metrics, 'EuclideanDistance', EuclideanDistance64)
                
                # Update in sys.modules - CRITICAL for unpickling
                mod_name = 'sklearn.metrics._dist_metrics'
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], 'EuclideanDistance', EuclideanDistance64)
                
                if hasattr(dist_metrics, '__dict__'):
                    dist_metrics.__dict__['EuclideanDistance'] = EuclideanDistance64
                
                print("[OK] Patched EuclideanDistance using EuclideanDistance64")
            
            # Fallback: Use EuclideanDistance32
            elif hasattr(dist_metrics, 'EuclideanDistance32'):
                EuclideanDistance32 = dist_metrics.EuclideanDistance32
                dist_metrics.EuclideanDistance = EuclideanDistance32
                setattr(dist_metrics, 'EuclideanDistance', EuclideanDistance32)
                
                mod_name = 'sklearn.metrics._dist_metrics'
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], 'EuclideanDistance', EuclideanDistance32)
                
                if hasattr(dist_metrics, '__dict__'):
                    dist_metrics.__dict__['EuclideanDistance'] = EuclideanDistance32
                
                print("[OK] Patched EuclideanDistance using EuclideanDistance32")
        
        # Ensure patch is in sys.modules
        if 'sklearn.metrics._dist_metrics' in sys.modules and hasattr(dist_metrics, 'EuclideanDistance'):
            if not hasattr(sys.modules['sklearn.metrics._dist_metrics'], 'EuclideanDistance'):
                setattr(sys.modules['sklearn.metrics._dist_metrics'], 'EuclideanDistance', dist_metrics.EuclideanDistance)
        
    except Exception as patch_error:
        print(f"Warning: Could not apply pre-patch: {patch_error}")
        import traceback
        traceback.print_exc()
    
    # Now try to load the model
    try:
        print("Loading model...")
        import joblib
        
        # Try standard loading first
        try:
            model = joblib.load('best_lung_cancer_model.joblib')
            scaler = joblib.load('scaler.joblib')
            print("[OK] Model and scaler loaded successfully!")
        except (AttributeError, ModuleNotFoundError, KeyError) as e:
            if 'EuclideanDistance' in str(e) or 'EuclideanDistance' in repr(e):
                print("Compatibility issue detected. Trying alternative loading method...")
                
                # Try using the model_loader
                try:
                    from model_loader import load_sklearn_model_safe
                    model, scaler = load_sklearn_model_safe('best_lung_cancer_model.joblib', 'scaler.joblib')
                    print("[OK] Model and scaler loaded successfully using compatibility loader!")
                except Exception as e2:
                    print(f"Compatibility loader also failed: {e2}")
                    raise e  # Raise original error
            else:
                raise
        
        # Print model info if available
        if hasattr(model, 'feature_names_in_'):
            print(f"Model expects {len(model.feature_names_in_)} features")
            print(f"Features: {list(model.feature_names_in_)}")
        if hasattr(model, 'classes_'):
            print(f"Model classes: {model.classes_}")
        if scaler and hasattr(scaler, 'n_features_in_'):
            print(f"Scaler expects {scaler.n_features_in_} features")
            
    except Exception as e:
        error_msg = str(e)
        print("\n" + "="*70)
        print("MODEL LOADING ERROR")
        print("="*70)
        print(f"\nError: {error_msg}")
        print("\nTroubleshooting steps:")
        print("\n1. Try installing a compatible scikit-learn version:")
        print("   pip uninstall scikit-learn")
        print("   pip install scikit-learn==1.2.2")
        print("\n2. If that doesn't work, try using Python 3.10 or 3.11")
        print("   (Python 3.12 may have compatibility issues)")
        print("\n3. Alternative: Install scikit-learn with pre-built wheels:")
        print("   pip install --only-binary :all: scikit-learn==1.2.2")
        print("\n4. Check that both model files exist:")
        print("   - best_lung_cancer_model.joblib")
        print("   - scaler.joblib")
        print("="*70 + "\n")
        import traceback
        traceback.print_exc()
        model = None
        scaler = None
        
except Exception as e:
    print(f"Critical error during initialization: {e}")
    import traceback
    traceback.print_exc()
    model = None
    scaler = None


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Request model for lung cancer prediction.
    """
    gender: str = Field(..., description="Patient gender", examples=["M"])
    age: float = Field(..., ge=1, le=150, description="Patient age", examples=[65])
    smoking: str = Field(..., description="Smoking status", examples=["YES"])
    yellow_fingers: str = Field(..., description="Yellow fingers symptom", examples=["NO"])
    anxiety: str = Field(..., description="Anxiety symptom", examples=["NO"])
    peer_pressure: str = Field(..., description="Peer pressure", examples=["NO"])
    chronic_disease: str = Field(..., description="Chronic disease", examples=["YES"])
    fatigue: str = Field(..., description="Fatigue symptom", examples=["YES"])
    allergy: str = Field(..., description="Allergy", examples=["NO"])
    wheezing: str = Field(..., description="Wheezing symptom", examples=["YES"])
    alcohol: str = Field(..., description="Alcohol consumption", examples=["NO"])
    coughing: str = Field(..., description="Coughing symptom", examples=["YES"])
    shortness_of_breath: str = Field(..., description="Shortness of breath", examples=["YES"])
    swallowing_difficulty: str = Field(..., description="Swallowing difficulty", examples=["NO"])
    chest_pain: str = Field(..., description="Chest pain symptom", examples=["YES"])
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate gender is M or F."""
        v = v.upper()
        if v not in ['M', 'F']:
            raise ValueError('gender must be "M" or "F"')
        return v
    
    @field_validator('smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 
                     'chronic_disease', 'fatigue', 'allergy', 'wheezing', 
                     'alcohol', 'coughing', 'shortness_of_breath', 
                     'swallowing_difficulty', 'chest_pain')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate YES/NO fields."""
        v = v.upper()
        if v not in ['YES', 'NO']:
            raise ValueError('must be "YES" or "NO"')
        return v


class PredictionResponse(BaseModel):
    """
    Response model for prediction.
    """
    success: bool = Field(..., description="Indicates if prediction was successful")
    prediction: str = Field(..., description="Prediction result: YES or NO")
    probability: float = Field(..., description="Confidence percentage")
    message: str = Field(..., description="Human-readable message")


class StatusResponse(BaseModel):
    """
    Response model for status endpoint.
    """
    status: str = Field(..., description="API status message")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/",
    summary="API Root",
    description="Root endpoint with API information",
    tags=["Info"]
)
async def root():
    """Root endpoint that provides API information."""
    return {
        "message": "Welcome to the Lung Cancer Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "GET /status": "Check API status",
            "POST /predict": "Predict lung cancer risk"
        }
    }


@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Check API Status",
    description="Returns the current status of the API and model loading status",
    tags=["Health"]
)
async def get_status():
    """
    Health check endpoint.
    
    Returns:
        StatusResponse: Status message indicating if API and model are ready
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler not loaded"
        )
    
    return StatusResponse(status="API is running and model is loaded")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Lung Cancer Risk",
    description="Predict lung cancer risk based on patient symptoms and characteristics",
    tags=["Prediction"]
)
async def predict(data: PredictionRequest):
    """
    Predict lung cancer risk based on patient data.
    
    Args:
        data: PredictionRequest containing patient information
        
    Returns:
        PredictionResponse: Prediction result with confidence score
        
    Raises:
        HTTPException: 500 if model not loaded, 400 if validation fails
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model or scaler not loaded. Please check server logs for details."
        )
    
    try:
        # Convert YES/NO to numeric (YES=2, NO=1)
        smoking = 2 if data.smoking == 'YES' else 1
        yellow_fingers = 2 if data.yellow_fingers == 'YES' else 1
        anxiety = 2 if data.anxiety == 'YES' else 1
        peer_pressure = 2 if data.peer_pressure == 'YES' else 1
        chronic_disease = 2 if data.chronic_disease == 'YES' else 1
        fatigue = 2 if data.fatigue == 'YES' else 1
        allergy = 2 if data.allergy == 'YES' else 1
        wheezing = 2 if data.wheezing == 'YES' else 1
        alcohol = 2 if data.alcohol == 'YES' else 1
        coughing = 2 if data.coughing == 'YES' else 1
        shortness_of_breath = 2 if data.shortness_of_breath == 'YES' else 1
        swallowing_difficulty = 2 if data.swallowing_difficulty == 'YES' else 1
        chest_pain = 2 if data.chest_pain == 'YES' else 1
        
        # Try different gender encodings
        # Pattern 1: M=1, F=0 (binary)
        gender_encoded = 1 if data.gender == 'M' else 0
        
        # Create feature array
        features_v1 = np.array([[
            gender_encoded,  # Gender: M=1, F=0
            data.age,
            smoking,
            yellow_fingers,
            anxiety,
            peer_pressure,
            chronic_disease,
            fatigue,
            allergy,
            wheezing,
            alcohol,
            coughing,
            shortness_of_breath,
            swallowing_difficulty,
            chest_pain
        ]], dtype=np.float64)
        
        # Try alternative: gender as M=2, F=1
        gender_encoded_v2 = 2 if data.gender == 'M' else 1
        features_v2 = np.array([[
            gender_encoded_v2,  # Gender: M=2, F=1
            data.age,
            smoking,
            yellow_fingers,
            anxiety,
            peer_pressure,
            chronic_disease,
            fatigue,
            allergy,
            wheezing,
            alcohol,
            coughing,
            shortness_of_breath,
            swallowing_difficulty,
            chest_pain
        ]], dtype=np.float64)
        
        # Try to make prediction with first encoding
        try:
            features_scaled = scaler.transform(features_v1)
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
        except:
            # If that fails, try second encoding
            try:
                features_scaled = scaler.transform(features_v2)
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error processing features: {str(e)}"
                )
        
        # Get probability and result
        # Model classes are [0, 1] where 0=NO, 1=YES
        if prediction == 1:
            result = "YES"
            probability = prediction_proba[1] * 100 if len(prediction_proba) > 1 else (1 - prediction_proba[0]) * 100
        else:
            result = "NO"
            probability = prediction_proba[0] * 100
        
        return PredictionResponse(
            success=True,
            prediction=result,
            probability=round(probability, 2),
            message=f'Prediction: {result} (Confidence: {probability:.2f}%)'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Prediction failed: {str(e)}'
        )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for validation errors."""
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": errors,
            "status_code": 422
        }
    )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    # Get port from environment variable (for deployment) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # --reload enables auto-reload on code changes (development only)
    reload = os.environ.get("ENVIRONMENT", "development") == "development"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload
    )
