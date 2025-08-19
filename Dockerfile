FROM python:3.11-slim

# Force rebuild - Updated 2025-08-18
ENV REBUILD_DATE=2025-08-18

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5001

# Run the API
CMD ["python", "api.py"]