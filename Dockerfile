FROM python:3.10

# Create a non-root user
RUN useradd -m appuser

# Set working directory
COPY . .

# Copy only requirements first (to leverage Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files, and assign to non-root user
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
