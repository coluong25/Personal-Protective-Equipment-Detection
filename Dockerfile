FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime dependencies (slim không có sẵn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libxcb1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng layer cache khi chỉ thay code
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Source code (không copy models/ — mount qua volume)
COPY src/  ./src/
COPY app/  ./app/
COPY data.yaml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
