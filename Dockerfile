# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port (Railway/Render will set PORT env variable)
EXPOSE 8000

# Run the application
# Railway/Render will override this with their start command that includes $PORT
# This is a fallback for local Docker runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

