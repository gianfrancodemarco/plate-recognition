# syntax = docker/dockerfile:experimental
FROM python:3.9-slim

WORKDIR /app

COPY ./src ./src
COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./models ./models

# Libraries for OpenCV
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip 

# Install project as module
RUN python3 -m pip install .

# Install dependencies using cache so that they don't have to be downloaded every time
RUN --mount=type=cache,target=/root/.cache pip install -r ./requirements.txt 

EXPOSE 80