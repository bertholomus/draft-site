FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir safetensors zstandard

WORKDIR /worker
COPY worker/client.py /worker/client.py
COPY worker/draft_worker.py /worker/draft_worker.py
COPY NOTICE /worker/NOTICE

ENTRYPOINT ["python3", "/worker/draft_worker.py"]
