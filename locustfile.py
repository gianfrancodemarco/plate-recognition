"""
Run load tests:
locust -f load_test/locustfile.py --host http://127.0.0.1:3000
"""

from locust import HttpUser, task

class WinePredictionUser(HttpUser):
    @task(1)
    def healthcheck(self):
        self.client.get("/docs")