import os

import uvicorn
from fastapi import FastAPI
from src.app.api import api
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "PUT", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(api.api_router, prefix="/api/v1")

if __name__ == "__main__":
    #port = (os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=8080)
