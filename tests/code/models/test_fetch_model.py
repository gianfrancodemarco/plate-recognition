import pytest
from src.models.fetch_model import fetch_model


class TestFetchModel():
    def test_fetch_model_error(self):
        with pytest.raises(ValueError):
            fetch_model()
