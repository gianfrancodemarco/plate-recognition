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
    def predict_bbox_as_image_false(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-bbox?as_image=false", files=files)

    @task(3)
    def predict_bbox_as_image_true(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-bbox?as_image=true", files=files)

    @task(5)
    def predict_bbox_annotate_image_postprocess_false(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-text?postprocess=false", files=files)

    @task(10)
    def predict_bbox_annotate_image_post_process_true(self):
        files = {'image_file': requests.get("https://picsum.photos/200").content}
        self.client.post("/api/v1/image-recognition/predict/plate-text?postprocess=true", files=files)
