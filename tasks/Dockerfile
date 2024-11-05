FROM python:3.12.4-slim-bookworm

WORKDIR /phen

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app/ ./app

COPY data/ ./data