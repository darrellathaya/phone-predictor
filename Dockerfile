# Use a lightweight base Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to allow external access
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
