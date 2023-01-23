import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from src.app.api import api
from src.app.monitoring import instrumentator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "PUT", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(api.api_router, prefix="/api/v1")
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
