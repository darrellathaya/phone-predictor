# Use appropriate base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies (adjust as needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
