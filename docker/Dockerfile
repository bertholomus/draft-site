FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 draftuser \
    && useradd --uid 10001 --gid 10001 --shell /usr/sbin/nologin --no-create-home draftuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server /app/server
COPY NOTICE /app/NOTICE
COPY LICENSE /app/LICENSE

RUN chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 8742

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8742/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8742", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
