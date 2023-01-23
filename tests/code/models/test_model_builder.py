from src.models.model_builder import ModelBuilder
from keras.models import Sequential

class TestModelBuilder():
    def test_build_model_no_params(self):
        model = ModelBuilder().build()
        assert isinstance(model, Sequential)
        assert len(model.layers) == 7

    def test_build_model(self):
        model = ModelBuilder(
            dropout = 0.5,
            cnn_blocks = 2,
            filters_num = 16,
            filters_kernel_size = (3, 3)
        ).build()
        assert isinstance(model, Sequential)
        assert len(model.layers) == 10