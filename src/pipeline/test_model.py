import os

import dvc.api
import mlflow
import numpy as np
from PIL import Image
from src import utils
from src.data.image_preprocessing import crop_image
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.features.postprocessing import post_process_plate
from src.models.fetch_model import fetch_model
from src.models.metrics import lev_dist
from src.pipeline.param_parser import ParamParser
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DATASETS_BASE = os.path.join(utils.DATA_PATH, "processed")

params = dvc.api.params_show()
params_dict = dvc.api.params_show()
params = ParamParser().parse(params_dict)


model = fetch_model(model_name=params.test.model.model_name,
                    model_version=params.test.model.model_version)
transformer_processor = TrOCRProcessor.from_pretrained(params.test.model.tr_ocr_processor)
transformer_model = VisionEncoderDecoderModel.from_pretrained(params.test.model.tr_ocr_model)


def evaluate_bbox_detection():
    [loss, root_mean_squared_error] = model.evaluate(test_set_bbox)

    mlflow.log_metrics({
        "loss": loss,
        "root_mean_squared_error": root_mean_squared_error
    })


def evaluate_ocr():

    bboxes = model.predict(test_set_bbox, batch_size=16)

    _accuracy, _accuracy_post_processed, _lev_dist, _lev_dist_post_processed = 0, 0, 0, 0

    for (bbox, sample) in zip(bboxes, test_set_plates):
        image = sample[0][0].numpy().astype(np.uint8)
        plate = sample[1][0].numpy().decode()
        
        try:
            cropped_image = crop_image(image, bbox)
            cropped_image = Image.fromarray(cropped_image)
            pixel_values = transformer_processor(cropped_image, return_tensors="pt").pixel_values
            generated_ids = transformer_model.generate(pixel_values)
            generated_text = transformer_processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]
        except:
            generated_text = ""

        _accuracy += 1 if generated_text == plate else 0
        _lev_dist += lev_dist(generated_text, plate)

        generated_text = post_process_plate(generated_text)

        _accuracy_post_processed += 1 if generated_text == plate else 0
        _lev_dist_post_processed += lev_dist(generated_text, plate)

    n_samples = len(bboxes)
    mlflow.log_metrics({
        "accuracy": _accuracy/n_samples,
        "accuracy_post_processed": _accuracy_post_processed/n_samples,
        "lev_dist": _lev_dist/n_samples,
        "lev_dist_post_processed": _lev_dist_post_processed/n_samples
    })


if __name__ == "__main__":

    dataset_generator_type = ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR
    if params.augmentation:
        dataset_generator_type = ImageDatasetType.BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR

    test_set_bbox = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "test", "annotations.csv"),
        dataset_generator_type=dataset_generator_type, batch_size=1, shuffle=False
    )
    
    test_set_plates = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "test", "annotations.csv"),
        dataset_generator_type=ImageDatasetType.PLATE_IMAGES_DATASET_GENERATOR, batch_size=1, shuffle=False
    )
    
    mlflow.set_experiment("Test")
    run_name = f"test_{params.test.model.model_name}_v{params.test.model.model_version}"
    with mlflow.start_run(run_name=run_name):
        evaluate_bbox_detection()
        evaluate_ocr()
