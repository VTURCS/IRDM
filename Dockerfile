FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

# CPU-only PyTorch first: the default Linux wheel pulls in several GB of GPU libraries.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Download the pinned model at build time, so a running container never needs the network.
COPY irdm.py .
RUN python -c "import irdm; irdm.load_model()"
ENV HF_HUB_OFFLINE=1

COPY . .
EXPOSE 8000
CMD ["flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "8000"]
