# Lung Cancer Prediction API

A FastAPI-based REST API for predicting lung cancer risk based on patient symptoms and characteristics.

## Features

- ✅ RESTful API endpoints
- ✅ Automatic Swagger/OpenAPI documentation
- ✅ Pydantic models for request validation
- ✅ CORS support for web applications
- ✅ Production-ready with error handling

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Deployment

See `QUICK_DEPLOY.md` for quick deployment instructions (Railway recommended).

For detailed deployment options, see `DEPLOYMENT.md`.

## API Endpoints

- `GET /` - API information
- `GET /status` - Check API status
- `POST /predict` - Predict lung cancer risk

## Request Format

```json
{
  "gender": "M",
  "age": 65,
  "smoking": "YES",
  "yellow_fingers": "NO",
  "anxiety": "NO",
  "peer_pressure": "NO",
  "chronic_disease": "YES",
  "fatigue": "YES",
  "allergy": "NO",
  "wheezing": "YES",
  "alcohol": "NO",
  "coughing": "YES",
  "shortness_of_breath": "YES",
  "swallowing_difficulty": "NO",
  "chest_pain": "YES"
}
```

## Response Format

```json
{
  "success": true,
  "prediction": "YES",
  "probability": 87.5,
  "message": "Prediction: YES (Confidence: 87.50%)"
}
```

## Notes

- This application is for educational/research purposes only
- Medical predictions should always be verified by healthcare professionals
- The model accuracy depends on the quality of the training data
