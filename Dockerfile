FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output dir
RUN mkdir -p /data/output

ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/data/output

ENTRYPOINT ["python", "main.py"]
