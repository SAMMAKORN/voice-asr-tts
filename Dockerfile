FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# sounddevice imports PortAudio at runtime, including in web mode.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN useradd --system --uid 10001 --create-home appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/', timeout=3)"

CMD ["sh", "-c", "exec python -m web.server --host 0.0.0.0 --port ${PORT:-8000}"]
