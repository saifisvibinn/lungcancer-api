# FastAPI Backend - User Registration API

A clean, production-ready FastAPI backend with user registration functionality.

## Features

- ✅ RESTful API endpoints
- ✅ Automatic Swagger/OpenAPI documentation
- ✅ Pydantic models for request validation
- ✅ Age validation (18+)
- ✅ Clean, readable, production-ready code
- ✅ Comprehensive error handling
- ✅ Type hints throughout

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

### Development Mode (with auto-reload)
```bash
uvicorn main:app --reload
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. GET /status
Check API status.

**Response:**
```json
{
  "status": "API is running"
}
```

**Example:**
```bash
curl http://localhost:8000/status
```

### 2. POST /register
Register a new user.

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "age": 25
}
```

**Success Response (201 Created):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "age": 25
  }
}
```

**Error Response (400 Bad Request):**
```json
{
  "success": false,
  "error": "User must be at least 18",
  "status_code": 400
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "age": 25
  }'
```

## Validation Rules

- **name**: Required, 1-100 characters, cannot be empty
- **email**: Required, must be valid email format
- **age**: Required, must be 18 or older

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

### Test with cURL

**Status endpoint:**
```bash
curl http://localhost:8000/status
```

**Register endpoint (valid):**
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Jane Smith", "email": "jane@example.com", "age": 25}'
```

**Register endpoint (invalid age):**
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Young User", "email": "young@example.com", "age": 16}'
```

### Test with Python

```python
import requests

# Test status endpoint
response = requests.get("http://localhost:8000/status")
print(response.json())

# Test register endpoint
response = requests.post(
    "http://localhost:8000/register",
    json={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 25
    }
)
print(response.json())
```

## Project Structure

```
.
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── FASTAPI_README.md    # This file
```

## Code Quality

- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Pydantic models for validation
- ✅ Proper HTTP status codes
- ✅ Error handling
- ✅ Clean, readable code structure
- ✅ Production-ready patterns

## Next Steps

To make this production-ready, consider adding:

1. **Database Integration**: Store users in a database (PostgreSQL, MongoDB, etc.)
2. **Authentication**: Add JWT or OAuth2 authentication
3. **Password Hashing**: If adding passwords, use bcrypt or similar
4. **Email Verification**: Send confirmation emails
5. **Rate Limiting**: Prevent abuse
6. **Logging**: Add structured logging
7. **Testing**: Add unit and integration tests
8. **Docker**: Containerize the application
9. **Environment Variables**: Use .env for configuration
10. **CORS**: Configure CORS if needed for frontend integration

## License

This project is provided as-is for educational purposes.

