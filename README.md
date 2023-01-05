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
