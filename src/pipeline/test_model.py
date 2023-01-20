import logging

import dvc.api
import mlflow
import numpy as np
from PIL import Image
from src.data.image_preprocessing import crop_image
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.features.postprocessing import post_process_plate
from src.models.fetch_model import fetch_model
from src.models.metrics import lev_dist
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

params = dvc.api.params_show()

assert "train" in params, "Required param train"
test_params = params["test"]

model_params = test_params.get("model")
assert "model_name" in model_params, "Required param model.model_name"
assert "model_version" in model_params, "Required param model.model_version"
model_name = model_params["model_name"]
model_version = model_params["model_version"]

assert "tr_ocr_processor" in model_params, "Required param model.tr_ocr_processor"
assert "tr_ocr_model" in model_params, "Required param model.tr_ocr_model"

tr_ocr_processor = model_params["tr_ocr_processor"]
tr_ocr_model = model_params["tr_ocr_model"]

model = fetch_model(model_name=model_name, model_version=model_version)


def evaluate_bbox_detection():
    [loss, root_mean_squared_error] = model.evaluate(test_set_bbox)

    mlflow.log_metrics({
        "loss": loss,
        "root_mean_squared_error": root_mean_squared_error
    })


def evaluate_ocr():

    transformer_processor = TrOCRProcessor.from_pretrained(tr_ocr_processor)
    transformer_model = VisionEncoderDecoderModel.from_pretrained(tr_ocr_model)
    bboxes = model.predict(test_set_bbox, batch_size=16)

    _accuracy, _accuracy_post_processed, _lev_dist, _lev_dist_post_processed = 0, 0, 0, 0

    for (bbox, sample) in zip(bboxes, test_set_plates):
        image = sample[0][0].numpy().astype(np.uint8)
        plate = sample[1][0].numpy().decode()
        cropped_image = crop_image(image, bbox)
        cropped_image = Image.fromarray(cropped_image)
        pixel_values = transformer_processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = transformer_model.generate(pixel_values)
        generated_text = transformer_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

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
    test_set_bbox = get_dataset(
        "test", dataset_generator_type=ImageDatasetType.BboxImagesDatasetGenerator, batch_size=1, shuffle=False)
    test_set_plates = get_dataset(
        "test", dataset_generator_type=ImageDatasetType.PlateImagesDatasetGenerator, batch_size=1, shuffle=False)

    run_name = f"test_{model_name}_v{model_version}"
    with mlflow.start_run(run_name=run_name):
        evaluate_bbox_detection()
        evaluate_ocr()
