[![Code checks (pylint, pynblint, pytest & coverage](https://github.com/gianfrancodemarco/plate-recognition/actions/workflows/code_checks_plate_recognition_app.yml/badge.svg)](https://github.com/gianfrancodemarco/plate-recognition/actions/workflows/code_checks_plate_recognition_app.yml)
[![API endpoint](https://img.shields.io/badge/endpoint-API-blue?logo=googlechrome&logoColor=white)](https://plate-recognition-qbly4ubf5q-uc.a.run.app/docs#/)
[![Telegram BOT](https://img.shields.io/badge/endpoint-UI-blue?logo=googlechrome&logoColor=white)](https://t.me/PlateRecognitionBOT)

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
