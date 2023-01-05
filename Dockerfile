# syntax = docker/dockerfile:experimental
FROM python:3.9-slim

COPY ./src ./src
COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./models ./models

RUN python3 -m pip install --upgrade pip 
RUN python3 -m pip install . 
# Install dependencies using cache so that they don't have to be downloaded every time
RUN --mount=type=cache,target=/root/.cache/pip pip install -r ./requirements.txt 

EXPOSE 80