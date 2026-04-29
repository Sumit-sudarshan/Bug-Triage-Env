FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for Hugging Face Spaces (Standard)
EXPOSE 7860

# Start the OpenEnv server
CMD ["python", "-m", "server.app"]
