from src.models.model_builder import build_model
from keras.models import Sequential

class TestBuildModel():
    def test_build_model_no_params(self):
        model = build_model()
        assert isinstance(model, Sequential)
        assert len(model.layers) == 7

    def test_build_model(self):
        model = build_model(
            dropout = 0.5,
            cnn_blocks = 2,
            filters_num = 16,
            filters_kernel_size = (3, 3)
        )
        assert isinstance(model, Sequential)
        assert len(model.layers) == 10