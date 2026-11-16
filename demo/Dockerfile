FROM python:3.11-slim

# Thư viện cho OpenCV mở cửa sổ (X11) và đọc webcam
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY hand_localization_model.h5 .
COPY demo/test_model.py demo/test_model.py

ENTRYPOINT ["python", "demo/test_model.py"]
