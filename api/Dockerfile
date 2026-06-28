FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and api code
COPY models/ /app/models/
COPY api/ /app/api/

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]