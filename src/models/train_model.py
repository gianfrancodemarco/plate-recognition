import dvc.api
import mlflow
from src.features.dataset import get_dataset
from src.models.get_model import get_model
from src.models.model_trainer import train_model

params = dvc.api.params_show()
train_params = params['train']

assert 'epochs' in train_params, 'Required param epochs'
assert 'model_name' in train_params, 'Required param model_name'
assert 'model_version' in train_params, 'Required param model_version'

if __name__ == "__main__":

    train_set = get_dataset("train")
    validation_set = get_dataset("validation")

    model = get_model(
        dropout=train_params.get('dropout'),
        model_name=train_params['model_name'],
        model_version=train_params['model_version']
    )

    with mlflow.start_run():

        mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_models=True
        )

        model = train_model(
            model=model,
            dataset=train_set,
            validation_dataset=validation_set,
            epochs=train_params['epochs'],
            model_name=train_params['model_name'],
            save_every_n_epochs=train_params['save_every_n_epochs']
        )
