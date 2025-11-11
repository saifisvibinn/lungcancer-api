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
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Railway/Render will set PORT env variable)
EXPOSE 8000

# Run the application using the startup script
# This properly handles the PORT environment variable
# Using shell form to execute the bash script
CMD ./start.sh

