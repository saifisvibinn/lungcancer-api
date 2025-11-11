#!/bin/bash
# Startup script for Railway deployment
# Reads PORT from environment variable and starts uvicorn

PORT=${PORT:-8000}
uvicorn main:app --host 0.0.0.0 --port $PORT

