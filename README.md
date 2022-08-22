DATASETS 

Only images:
baza_slika: http://www.zemris.fer.hr/projects/LicensePlates/english/

Images and bounding boxes:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download

Romanian Plates:
https://github.com/RobertLucian/license-plate-dataset


INSTRUCTIONS

1) Download datasets and extract them in datasets/
   a) For license-plate-recognition, unify train and validation under a single folder
2) Run src/preprocessing/preprocess.py. This should create src/resources directory and populate it with resized images from the datasets
3) Run src/preprocessing/create_dataset.py. This will create the folder /final_dataset with all of the images and annotations merged from the various datasets

At this point, the images were reviewed one by one, and those were the plates was not clearly visible were removed.
In particular most of the license-plate-dataset were removed, because the resizing scaled them a lot.

At this point, a human in the loop process is started:
1) Train a model to predict the bounding box on the current dataset (images + annotations)
2) Predict the bounding box for all of the images without a label
3) Show the images with the predictions; if it is good enough, save the annotation and move it into the dataset.
Even if the bounding box is not very precise, if it contains all of the text it can be approved; larger bounding box than the plate are ok,
smaller obviously not.
4) Repeated from one until there are no more images without label.





1) Resize a allineamento annotations
2) Rimozione di quelle che con il resize non si leggevano più (maggior parte di license-plate-recognition)
3) Human-in-the-loop
4) Rimozione di tutte tranne baza_slika

FINE TUNED MODELS: 200MB
CUSTOM MODELS: 10MB