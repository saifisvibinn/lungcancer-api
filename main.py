"""
FastAPI Backend Application
A simple REST API with user registration functionality.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, Field, field_validator
import uvicorn
import os

# Initialize FastAPI application
# This automatically enables Swagger UI at /docs and ReDoc at /redoc
app = FastAPI(
    title="User Registration API",
    description="A simple API for user registration with validation",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc"  # Alternative API documentation
)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class RegisterRequest(BaseModel):
    """
    Request model for user registration.
    Validates name, email, and age fields.
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's full name",
        examples=["John Doe"]
    )
    email: EmailStr = Field(
        ...,
        description="User's email address",
        examples=["john.doe@example.com"]
    )
    age: int = Field(
        ...,
        description="User's age (must be 18 or older)",
        examples=[25]
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Validate that age is 18 or older."""
        if v < 18:
            raise ValueError("User must be at least 18")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "age": 25
            }
        }


class RegisterResponse(BaseModel):
    """
    Response model for successful registration.
    """
    success: bool = Field(
        ...,
        description="Indicates if registration was successful",
        examples=[True]
    )
    message: str = Field(
        ...,
        description="Confirmation message",
        examples=["User registered successfully"]
    )
    user: dict = Field(
        ...,
        description="Registered user information",
        examples=[{
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 25
        }]
    )


class StatusResponse(BaseModel):
    """
    Response model for status endpoint.
    """
    status: str = Field(
        ...,
        description="API status message",
        examples=["API is running"]
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Check API Status",
    description="Returns the current status of the API",
    tags=["Health"]
)
async def get_status():
    """
    Health check endpoint.
    
    Returns:
        JSONResponse: Status message indicating the API is running
        
    Example Response:
        {
            "status": "API is running"
        }
    """
    return StatusResponse(status="API is running")


@app.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a New User",
    description="Register a new user with name, email, and age. Age must be 18 or older.",
    tags=["Users"]
)
async def register_user(user_data: RegisterRequest):
    """
    Register a new user endpoint.
    
    This endpoint accepts user registration data and validates:
    - Name: Must be non-empty string (1-100 characters)
    - Email: Must be a valid email format
    - Age: Must be 18 or older
    
    Args:
        user_data (RegisterRequest): User registration data
        
    Returns:
        RegisterResponse: Success confirmation with user data
        
    Raises:
        HTTPException: 400 Bad Request if validation fails
        HTTPException: 422 Unprocessable Entity if request format is invalid
        
    Example Request:
        {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 25
        }
        
    Example Response:
        {
            "success": true,
            "message": "User registered successfully",
            "user": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "age": 25
            }
        }
    """
    # Age validation is handled by Pydantic field_validator
    # In a real application, you would:
    # 1. Check if email already exists in database
    # 2. Hash password if included
    # 3. Save user to database
    # 4. Send confirmation email
    # For now, we'll just return a success response
    
    return RegisterResponse(
        success=True,
        message="User registered successfully",
        user={
            "name": user_data.name,
            "email": user_data.email,
            "age": user_data.age
        }
    )


# ============================================================================
# Custom Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions.
    Returns consistent error response format.
    """
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
    """
    Custom handler for Pydantic validation errors.
    Converts validation errors to 400 Bad Request with custom message for age.
    """
    errors = exc.errors()
    
    # Check if the error is related to age validation
    for error in errors:
        error_loc = error.get("loc", [])
        error_msg = str(error.get("msg", ""))
        
        # Check if this is an age validation error
        if "age" in error_loc and ("User must be at least 18" in error_msg or "18" in error_msg):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "User must be at least 18",
                    "status_code": 400
                }
            )
    
    # For other validation errors, return standard format
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
    # Run the application using uvicorn
    # Get port from environment variable (for deployment) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # --reload enables auto-reload on code changes (development only)
    # In production, reload should be False
    reload = os.environ.get("ENVIRONMENT", "development") == "development"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload
    )

