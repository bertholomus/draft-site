FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir safetensors==0.5.3 zstandard==0.23.0 tokenizers==0.21.1

WORKDIR /worker
COPY worker/client.py /worker/client.py
COPY worker/label_worker.py /worker/label_worker.py
COPY worker/draft_worker.py /worker/draft_worker.py
COPY NOTICE /worker/NOTICE
COPY LICENSE /worker/LICENSE

ENTRYPOINT ["python3", "/worker/label_worker.py"]
