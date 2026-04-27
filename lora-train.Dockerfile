FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 gcc curl && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /sd-scripts
WORKDIR /sd-scripts

# Install from requirements + onnxruntime for WD14 tagger
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir accelerate onnxruntime-gpu onnx

# Copy training API server
COPY train/server.py /train-server.py

WORKDIR /workspace
EXPOSE 8787

# Default: run the HTTP API server
# Override CMD to run one-shot training (e.g. accelerate launch ...)
CMD ["python", "/train-server.py"]
