FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, entrypoint, and model
COPY src/ /app/src
COPY model.pt /app/model.pt


EXPOSE 8000

CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
