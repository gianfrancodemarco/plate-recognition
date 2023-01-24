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
    1. [Dagshub](#dagshub)
    2. [DVC](#dvc)
        1. [Pipelines](#pipelines)
    3. [MLFlow](#mlflow)
    4. [Random Number Generator](#random-number-generator)
    5. [.env](#env)
    6. [Github variables and Secrets](#github-variables-and-secrets)
4. [Quality assurance](#quality-assurance)
    1. [Pylint](#pylint)
    2. [Pynblint](#pynblint)
    3. [Pytest and coverage](#pytest-and-coverage)
    4. [Great Expectations](#great-expectations)   
5. [API](#api)
6. [Telegram Bot](#telegram-bot)
7. [CI/CD](#cicd)
    1. [Code Checks](#code-checks)
    2. [DockerHub](#dockerhub)
    3. [Artifact Registry & Cloud Run](#artifact-registry-cloud-run)
8. [Monitoring](#monitoring)
9. [Extra](#extra)
    1. [Hyperparameters optimization](#optimization)
    2. [Project structure](#project-structure)
    3. [Security](#security)
    4. [Costs](#costs)
    5. [Developer Guide](#developer-guide)


# The project
This project aims to build a service that recognizes and transcribes the license plate of a vehicle from a picture.
The service is composed of:
- A [backend service](https://plate-recognition-qbly4ubf5q-uc.a.run.app/docs#/), which exposes the API that interact with the model
- A [Telegram bot](https://t.me/PlateRecognitionBOT), that can be used from the users to interact with the service
- A [Prometheus instance](https://prometheus-qbly4ubf5q-uc.a.run.app), for monitoring
- A [Grafana instance](https://grafana-qbly4ubf5q-uc.a.run.app), for monitoring and data analysis

The plate recognition is performed in two steps:
- plate detection: uses a custom model to find the bounding box of the plate 
- image-to-text: uses a pretrained model to transcribe the plate

# Inception
## Project Organization

The project initial organization has been created using the [cookiecutter data sicence template](https://drivendata.github.io/cookiecutter-data-science/) and then adapted to the needs of this projects.

For example, the folder `src/pipeline` has been added to gather all of the scripts composing the dvc pipeline.
Another example is the absence of the `model` folder, since this project is completely integrated with MLFlow and uses an the MLFlow server provided by Dagshub to store the models.

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

Describe the 2 models!

## Dataset card
# Reproducibility

## Dagshub

Dagshub is a Github's inspired platform, specifically created for data science projects, that allows to host, version, and manage code, data, models, experiments,
Dagshub is free and open-source. Furthermore, reaching out to the mantainers on their discord server, it is possible to obtain a free pro license, accorded to students.

Dagshub comes with a DVC remote storage (100GB for pro accounts) and a remote MLFlow server.


## DVC

DVC is a software, based on Git, that allows to version data and track data science experiments.

In this project, the contents of the `data` folder is stored and tracked using 
DVC.
The remote storage used is the one offered by Dagshub.

### Pipelines

DVC allows not only to version data, but also to create fully reproducible pipelines.
The pipelines are defined using the CLI or by manually editing the `dvc.yaml` file.

A pipelines of 5 steps has been defined:
- `download_data`: downloads the raw data from an external source
- `make_dataset`: reorganizes the raw data in a viable datasets, then splits the data in train, validation and test sets
- `data_expectations`: checks the structure of the data. This project is based on image data, which find no support from GreatExpectations. For this reason, only the annotations have been checked.
- `train_model`: trains the model, tracking the experiment and the artifacts on the remote MLFlow server
- `test_model`: tests the model on the test set and stores the results on the remote MLFlow server 

\
\
![The DVC pipeline](reports/figures/dvc_pipeline.png "DVC Pipeline")

The pipeline can be configured using the `params.yaml` file.
This file contains configurations for the structure of the model, the training and the testing phases.


### MLFlow

MLFLow is a software that allows to track Machine Learning experiments and models.
It stores the metrics of the experiments, allowing the developer to compare different models and parameters. Also, allows to store the models and retrieve them when needed.

This project is fully integrated with MLFlow:
- for the training and testing phase, experiments and models are trackend on the Dagshub's MLFlow server
- the API service loads the image detection model directly from Dagshub.

For this integration, the following environment variables are required to be set:

- MLFLOW_TRACKING_URI
- MLFLOW_TRACKING_USERNAME
- MLFLOW_TRACKING_PASSWORD

How to set these variables in dev and production environment is described in the next sections.

### Random number generator

In a data science project, it is not rare to rely on randomness to perform some tasks: dataset splitting in different sets, Neural Networks weight initialization, etc.
This puts a threath for the experiments reproducibility, since even if the developer doesn't perform any change in the proejct, some internal mechanisms will behave slightly different from one run to another.
To overcome this issue, it is a good practice to fix the seeds for the random generators used by the project.

For this project, this is done by the function `set_random_states` in `src/utils.py`, which is called at appropriate spots.
Moreover, the random state which is set can be changed in the `params.yaml` file.


### .env

It is a good practice to not place in the source code all of those configurations that can change in time, differ by environment, or that need to be classified.
For local environments, this is usually done using env files.

In these project, the `.env` file is read at the startup by the program, so that all of the environment variables are accessible by the code.
This file **MUST NOT** be added to version control (Git) or docker images, as it may contain API Tokens or passwords.
For this reason, when the services are executed in the context of `docker-compose`, the `.env` file is externally mounted into the services.


### Github variables and secrets

Since the `.env` file is not added to version control, these values must be stored somewhere to be accessible to the CI.

This is done using Github's:
- environment variables: here are stored all of those configuration which are not reserved, such as paths configurations. These are kept in clear
- secrets: here are stored all of the sensible configurations, like passwords and tokens. There are encrypted

The following have been defined:

**Environment variables**
- APP_DOCKER_IMAGE
- APP_SERVICE_NAME
- BOT_DOCKER_IMAGE
- BOT_SERVICE_NAME
- GRAFANA_DOCKER_IMAGE
- GRAFANA_SERVICE_NAME
- MODEL_NAME
- MODEL_VERSION
- PLATE_RECOGNITION_APP_HOST
- PLATE_RECOGNITION_APP_PORT
- PLATE_RECOGNITION_APP_SCHEMA
- PROMETHEUS_DOCKER_IMAGE
- PROMETHEUS_HOST
- PROMETHEUS_PORT
- PROMETHEUS_SCHEMA
- PROMETHEUS_SERVICE_NAME
- TELEGRAM_BOT_HOST
- TR_OCR_MODEL
- TR_OCR_PROCESSOR

**Secrets**
- CLOUD_RUN_JSON_KEY
- DAGSHUB_TOKEN
- DAGSHUB_USERNAME
- DOCKERHUB_TOKEN
- DOCKERHUB_USERNAME
- GAR_JSON_KEY
- MLFLOW_TRACKING_PASSWORD
- MLFLOW_TRACKING_URI
- MLFLOW_TRACKING_USERNAME
- TELEGRAM_API_TOKEN

# Quality assurance

## Pylint

This project integrates pylint, which is a static code analyser for Python, which checks the quality of the source code.

## Pynblint

This project integrates pynblint, which is a static code analyser for Python notebooks.

## Pytest and coverage

This project integrates pytest and coverage for unit testing of the code.
Pytest is a framework that allows to write and execute unit tests.

Coverage can be used as a wrapper for pytest, and is useful to produce statistics about the testing of the code (files/lines coverage).

## Great Expectations

Great expectations is a library that allows to check for the quality and charateristichs of the data. Specifically, it is useful to detect changes in the characteristics of the data in time.

Great expectations is mainly indicated for tabular data. Since this project is based mainly on image data, Great expectations has been used exclusively to test the images annotations.

# API

The plate recognition service is exposed using an HTTP server.
The framework used to build the APIs is FastAPI.

The service loads the two required models at startup:
- the plate recognition model (custom) from MLflow
- the image-to-text model (pre trained) from HuggingFace

The API exposed are:
```
URL: /api/v1/image-recognition/predict/plate-bbox
Body: 
- image_file (Binary)
Parameters:
- as_image (boolean)
    
Predicts the bbox of the plate for an image.
If as_image is true, it returns the original image with the bbox drawn in overlay; otherwise, it returns the array representing the bbox.
```

```
URL: /api/v1/image-recognition/predict/plate-text
Body: 
- image_file (Binary)
Parameters:
- postprocess (boolean)
    
Predicts the plate for an image.
If postprocess is true, it applies some postprocessing to the predicted plate.
```

# Telegram bot

The frontend chosen for these project is a Telegram Bot.
Telegram bots are user friendly and easy-to-setup solutions to provide interfaces for a service.

The telegram bot for this project can be found [here](https://t.me/PlateRecognitionBOT).

The user can send here an image, and the bot will respond with the predicted place.

The telegram bot backend is a stand-alone service, and sends an HTTP request to the `/api/v1/image-recognition/predict/plate-text` to predict the license plate.

It always uses the `postprocess` parameter as true.

![A sample conversation with the bot](reports/figures/bot_conversation.png "Conversation")
