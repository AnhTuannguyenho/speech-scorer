# Speech Scorer — RunPod Serverless LOAD BALANCER image (GPU).
# Chạy Flask HTTP server (app.py) trên PORT (mặc định 80), có /ping cho health check.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_NO_CACHE_DIR=1 \
    PORT=80

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg espeak-ng libsndfile1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install 'numpy<2' faster-whisper transformers flask soundfile phonemizer

COPY app.py /app/

# Nướng sẵn model
ARG ASR_MODEL=medium.en
ENV ASR_MODEL=${ASR_MODEL}
RUN python - <<'PY'
import os
os.environ["HF_HOME"] = "/models"
m = os.environ.get("ASR_MODEL", "medium.en")
from faster_whisper import WhisperModel
WhisperModel(m, device="cpu", compute_type="int8")
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
W = "facebook/wav2vec2-lv-60-espeak-cv-ft"
Wav2Vec2Processor.from_pretrained(W); Wav2Vec2ForCTC.from_pretrained(W)
print("models cached")
PY

ENV ASR_DEVICE=cuda ASR_COMPUTE=float16
EXPOSE 80
CMD ["python", "-u", "app.py"]
