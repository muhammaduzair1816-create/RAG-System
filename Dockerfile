# syntax=docker/dockerfile:1
#
# Image for Render (and any container host). Docker is used rather than Render's
# native Python runtime because OCR needs two system programs — Tesseract and
# Poppler — that pip cannot install and the native runtime cannot provide.
FROM python:3.12-slim

# HF_HOME fixes the model cache inside the image so the weights baked in at
# build time are the ones found at runtime. Render's free disk is ephemeral:
# without this the app would re-download every model after each spin-down.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    OMP_NUM_THREADS=1

# tesseract-ocr  -> reads scanned pages
# poppler-utils  -> pdf2image renders PDF pages for OCR
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements-render.txt omits sentence-transformers (and therefore torch),
# which would add ~2 GB to the image for no benefit: the default onnx backend
# runs the same checkpoint and returns identical vectors.
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

# Bake the embedding and speech models into the image. This keeps the first
# request after a cold start fast and avoids re-downloading ~130 MB every time
# the free instance wakes up.
RUN python scripts/prefetch_models.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

# Shell form so ${PORT}, which Render injects, is expanded.
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
