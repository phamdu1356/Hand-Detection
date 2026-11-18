# Drone demo image (CPU). The model weights are not in the image: mount them at runtime
# (see demo/docker-compose.yml, service "fly").
FROM python:3.11-slim

# build-essential is kept in case pybullet has to build from source; the X11 and OpenCV libraries
# let the HUD window open when DISPLAY is forwarded.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, so ultralytics does not pull the CUDA wheels.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY dronectl/ dronectl/
COPY sim/ sim/
COPY demo/ demo/
COPY tools/ tools/
# only the synthetic scenario helpers from tests/ are needed at runtime (fly_sim --synthetic)
COPY tests/synth.py tests/synth.py

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "demo/fly_sim.py"]
CMD ["--backend", "sim", "--no-gui", "--synthetic"]
