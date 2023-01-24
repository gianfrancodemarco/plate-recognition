from src.utils import dict2obj, rgetattr

class Params():
    def __init__(self, dict1):
        self.__dict__.update(dict1)

class ParamParser():

    """
    An array representing the required params in dot notation
    """
    REQUIRED_PARAMS = [
        "random_state",
        "augmentation",
        "train",
        "train.model",
        "train.model.model_name",
        "train.model.model_version",
        "train.model.dropout",
        "train.model.cnn_blocks",
        "train.model.filters_num",
        "train.model.filters_kernel_size",
        "train.model.optimizer_name",
        "train.model.learning_rate",
        "train.fit",
        "train.fit.epochs",
        "test",
        "test.model",
        "test.model.model_name",
        "test.model.model_version",
        "test.model.tr_ocr_processor",
        "test.model.tr_ocr_model",
    ]

    def parse(self, params_dict: dict) -> Params:
        """
        Builds a Params object and validates its structure

        :param dict params_dict: A dictionary representing the params
        """

        params: Params = dict2obj(params_dict, Params)
        self.__validate_params(params)
        return params

    def __validate_params(self, params: Params) -> Params:
        """
        Validates the structure of top level params.
        Throws an exception if required fields are missing
        """

        for param in self.REQUIRED_PARAMS:
            self.__validate_param(params, param)


    def __validate_param(self, params: Params, param_path: str) -> None:
        try:
            rgetattr(params, param_path)
        except AttributeError:
            raise KeyError(f"{param_path} is required")
