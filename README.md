# FastAPI User Registration Backend

A clean, production-ready FastAPI backend with user registration functionality.

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

## Documentation

- **Quick Start Guide:** `FASTAPI_README.md`
- **Deployment Guide:** `DEPLOYMENT.md`
- **Quick Deploy:** `QUICK_DEPLOY.md`

## API Endpoints

- `GET /status` - Check API status
- `POST /register` - Register a new user (requires name, email, age 18+)

## Features

- ✅ RESTful API endpoints
- ✅ Automatic Swagger/OpenAPI documentation
- ✅ Pydantic models for request validation
- ✅ Age validation (18+)
- ✅ Clean, production-ready code
