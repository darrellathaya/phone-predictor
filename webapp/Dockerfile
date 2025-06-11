# Use official Python base image
FROM python:3.11-slim

# Avoid buffering for logs
ENV PYTHONUNBUFFERED=1

# Extend timeout in Gunicorn (optional but helpful for Azure)
ENV GUNICORN_CMD_ARGS="--timeout 240"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Launch FastAPI with Gunicorn and Uvicorn workers
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]