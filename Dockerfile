# Dockerfile for BirdCLEF-2025 Training

FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-2.py310:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libsndfile1 \\
    libgl1 \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD ["birdclef_training.py"]
