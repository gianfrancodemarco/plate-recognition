DATASETS 


Only images:
baza_slika: http://www.zemris.fer.hr/projects/LicensePlates/english/

Images and bounding boxes:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download

Romanian Plates:
https://github.com/RobertLucian/license-plate-dataset

Images and bounding boxes and license plate number:
https://github.com/detectRecog/CCPD


INSTRUCTIONS

1) Download datasets and extract them in datasets/
   a) For license-plate-recognition, unify train and validation under a single folder
2) Run src/preprocessing/preprocess.py. This should create src/resources directory and populate it with resized images from the datasets
3) Run src/preprocessing/create_dataset.py. This will create the folder /final_dataset with all of the images and annotations merged from the various datasets