FROM python:3.12.2-slim-bookworm

WORKDIR /phen

RUN apt-get update

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app/ ./app

COPY data/ ./data