FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# torch has no GPU to use in this container (sentence-transformers runs
# MiniLM on CPU only) — installing it from the default index pulls the
# full CUDA build (several GB of unused binaries). The CPU-only wheel is
# functionally identical here and cuts the image from ~8.9GB to ~1.5-2GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8006

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8006"]
