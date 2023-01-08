# syntax = docker/dockerfile:experimental
FROM python:3.8-slim

WORKDIR /app

# Libraries for OpenCV
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip 

# Install dependencies using cache so that they don't have to be downloaded every time
COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r ./requirements.txt 

COPY ./src ./src
COPY ./models ./models

# Install project as module
COPY ./setup.py ./setup.py
RUN python3 -m pip install .

EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]