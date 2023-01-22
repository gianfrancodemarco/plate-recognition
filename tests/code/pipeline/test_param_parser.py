import pytest
from src.pipeline.param_parser import ParamParser, Params


class TestParamParser:

    def test_param_parser(self):
        dictionary = {
            "random_state": 12,
            "train": {
                "fit": {
                    "epochs": 10
                },
                "model":  {
                    "model_name": "test",
                    "model_version": 1
                }
            },
            "test": {
                "model": {
                    "model_name": "test",
                    "model_version": 1,
                    "tr_ocr_processor": "test",
                    "tr_ocr_model": 1
                }
            }
        }

        params = ParamParser().parse(dictionary)
        assert isinstance(params, Params)

    def test_param_parser_error(self):

        params_dict = {
            "random_state": 12,
            "train": {
                "model":  {
                    "ciao": 1
                }
            }
        }

        with pytest.raises(KeyError):
            ParamParser().parse(params_dict)
