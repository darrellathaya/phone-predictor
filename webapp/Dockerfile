# Use appropriate base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy app and model files
COPY app/ ./app/
COPY models/ ./models/

# Install dependencies (adjust as needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
