"""
Run load tests:
locust --host http://localhost:8080 --headless --run-time 10s
"""

import requests
from locust import HttpUser, task


class WinePredictionUser(HttpUser):
    @task(1)
    def healthcheck(self):
        self.client.get("/docs")

    @task(3)
    def predict_bbox(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-bbox", files=files)

    @task(3)
    def predict_bbox_annotate_image(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-bbox/annotate-image", files=files)
