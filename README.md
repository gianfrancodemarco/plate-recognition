[![Code checks (pylint, pynblint, pytest & coverage](https://github.com/gianfrancodemarco/plate-recognition/actions/workflows/code_checks_plate_recognition_app.yml/badge.svg)](https://github.com/gianfrancodemarco/plate-recognition/actions/workflows/code_checks_plate_recognition_app.yml)
[![API endpoint](https://img.shields.io/badge/endpoint-API-blue?logo=googlechrome&logoColor=white&label=Endpoint)](https://plate-recognition-qbly4ubf5q-uc.a.run.app/docs#/)
[![Telegram BOT](https://img.shields.io/badge/endpoint-UI-blue?logo=googlechrome&logoColor=white&label=Telegram%20BOT)](https://t.me/PlateRecognitionBOT)
[![Prometheus](https://img.shields.io/badge/style-Monitoring-green?logo=prometheus&logoColor=orange&label=Prometheus)](https://prometheus-qbly4ubf5q-uc.a.run.app)
[![Grafana](https://img.shields.io/badge/style-Monitoring-green?logo=grafana&logoColor=orange&label=Grafana&style=Monitoring)](https://grafana-qbly4ubf5q-uc.a.run.app)

## Table of Contents
1. [The project](#the-project)
2. [Inception](#inception)
    1. [Model card](#model-card)
    2. [Dataset card](#dataset-card)
    3. [Data augmentation](#data-augmentation)
3.  [Reproducibility](#reproducibility)
    1. [DVC](#dvc)
    2. [MLFlow](#mlflow)
    3. [Dagshub](#dagshub)
4. [Quality assurance](#quality-assurance)
    1. [Pylint](#pylint)
    2. [Coverage](#coverage)
    3. [Pynblint](#pynblint)
5. [API](#api)
6. [CI/CD](#cicd)
    1. [Code Checks](#code-checks)
    2. [DockerHub](#dockerhub)
    3. [Artifact Registry & Cloud Run](#artifact-registry-cloud-run)
7. [Monitoring](#monitoring)
8. [Extra](#extra)
    1. [Security](#security)
    2. [Costs](#costs)


# The-project
This project aims to build a service that recognizes and transcribes the license plate of a vehicle from a picture.
The service is composed of:
- A [backend service](https://plate-recognition-qbly4ubf5q-uc.a.run.app/docs#/), which exposes the API that interact with the model
- A [Telegram bot](https://t.me/PlateRecognitionBOT), that can be used from the users to interact with the service
- A [Prometheus instance](https://prometheus-qbly4ubf5q-uc.a.run.app), for monitoring
- A [Grafana instance](https://grafana-qbly4ubf5q-uc.a.run.app), for monitoring and data analysis

# Inception
## Project Organization
 ```
📦plate-recognition
 ┣ 📂.github
 ┃ ┗ 📂workflows            <- Workflow to be run on Github actions
 ┣ 📂data                    
 ┃ ┣ 📂external             <- Data from third party sources.
 ┃ ┣ 📂interim              <- Intermediate data that has been transformed.
 ┃ ┣ 📂processed            <- The final, canonical data sets for modeling.
 ┃ ┗ 📂raw                  <- The original, immutable data dump.
 ┣ 📂grafana
 ┣ 📂notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
 ┃                              the creator's initials, and a short `-` delimited description, e.g.
 ┃                              `1.0-jqp-initial-data-exploration`.
 ┣ 📂prometheus             <- Custom Docker image for prometheus
 ┣ 📂references             <- Data dictionaries, manuals, and all other explanatory materials.
 ┣ 📂reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
 ┃ ┣ 📂figures              <- Generated graphics and figures to be used in reporting
 ┃ ┣ 📂great_expectations
 ┃ ┣ 📂train
 ┣ 📂src                    <- Source code for use in this project.
 ┃ ┣ 📂app                     <- Source code for the APIs    
 ┃ ┃ ┣ 📂api
 ┃ ┣ 📂bot                     <- Source code for the Telegram Bot           
 ┃ ┣ 📂data                    <- Scripts to download or generate data
 ┃ ┣ 📂features                <- Scripts to turn raw data into features for modeling
 ┃ ┣ 📂models                  <- Scripts to train models and then use trained models to make predictions
 ┃ ┣ 📂pipeline                <- Script composing the experiment pipeline
 ┃ ┣ 📂visualization           <- Scripts to create exploratory and results oriented visualizations
 ┃ ┣ 📜logging.conf            <- Logging configuration
 ┣ 📂tests                  <- Python tests
 ┣ 📜.coverage
 ┣ 📜.dockerignore
 ┣ 📜.dvcignore
 ┣ 📜.gitignore
 ┣ 📜.pylintrc
 ┣ 📜Dockerfile_app         <- Dockerfile for the APIs
 ┣ 📜Dockerfile_bot         <- Dockerfile for the Telegram Bot
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜dev-requirements.txt
 ┣ 📜docker-compose.yaml    <- Docker compose file for local developement
 ┣ 📜dvc.lock
 ┣ 📜dvc.yaml
 ┣ 📜example.env            <- Example environment file
 ┣ 📜locustfile.py          <- Locust source code for load testing
 ┣ 📜params.yaml
 ┣ 📜requirements_app.txt
 ┣ 📜requirements_bot.txt
 ┣ 📜setup.py               <- makes project pip installable (pip install -e .) so src can be imported
 ┗ 📜tox.ini
```

## Model card
## Dataset card




## Commands

To run the main entry point, run:

```
python ./run.py 
```

To run a specific entry_point (e.g. download_data), run:
```
python ./run.py -e download_data 
```


<br/> 

## Pipeline

<br/>

### 1. Make dataset
Downloads the card database in SQLite format from MTGJson. <br/>
Then, for each card, downloads its image. <br/>
This step supports partial downloading of the resources and resuming.


### Start UVICORN local
python -m uvicorn src.app.main:app --reload 

### Start docker compose using modules cache
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose up

Labelling made with:
https://www.makesense.ai/
