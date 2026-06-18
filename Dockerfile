FROM python:3.12-slim

WORKDIR /app

# Install the CPU-only build of torch first (small, no CUDA), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
