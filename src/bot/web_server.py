"""
We need the container to be listening on a predefined port to be able to run it on CloudRun,
so we spawn a webserver along with the bot
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def healthz():
    return {"data": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8080
    )
