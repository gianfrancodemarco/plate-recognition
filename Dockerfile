FROM python:3.9-slim

COPY ./src ./src
COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./models ./models

RUN  python3 -m pip install --upgrade pip 
RUN python3 -m pip install . 
RUN python3 -m pip install -r ./requirements.txt

EXPOSE 80