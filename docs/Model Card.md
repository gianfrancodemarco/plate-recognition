---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for Plate Detection Model

<!-- Provide a quick summary of what the model is/does. -->

The model is a CNN trained on the [Plate Recognition Dataset](Dataset%20Card.md), which predicts the bounding box of the license plate in pascal voc* format (`xmin, ymin, xmax, ymax`).

# Model Details

## Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Gianfranco Demarco (gianfrademarco@gmail.com, g.demarco26@studenti.uniba.it)
- **Model type:** CNN
- **License:**  [MIT License](../LICENSE)

# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model should be used to perform detection of license plates on images.
## Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The output of this model can be passed as input to an image-to-text model to perform License Plate Recognition (transcription to text).
## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not intended to be used as a classifier of any sort.

# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model has been trained on the [Plate Recognition Dataset](Dataset%20Card.md).
This dataset is composed of images in very similar settings, so it won't perform as good as reported on different types of images.

# Training Details

## Training Data

This model has been trained on the [Plate Recognition Dataset](Dataset%20Card.md).
The dataset was split randomly in:
- training data (70%)
- validation data (10%)
- test data (20%)


## Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

### Params Optimization

The Optuna library was used to optimize the hyperparameter set described in [params.yaml](../params.yaml).
The optimizer was configured with a MedianPruner with `n_startup_trials=5, n_warmup_steps=10`.

### Preprocessing

The images must are resized to `256x256 pixels` before being fed into the model.
The images must have 3 color channels.
During the training process, the data has been augmented at runtime using the Albumentations library with the following settings:
```
transform = A.Compose([
    A.RandomCrop(p=0.3, width=200, height=200),
    A.HorizontalFlip(p=0.3),
    A.Blur(p=0.3),
    A.CLAHE(p=0.3),
    A.Equalize(p=0.3),
    A.ColorJitter(p=0.3),
    A.RandomShadow(p=0.3),
    A.RandomBrightness(p=0.3),
    A.ShiftScaleRotate(p=0.3, rotate_limit=15)
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=1, label_fields=[]))
```



### Shape
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 56)      1568      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 127, 127, 56)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 56)      28280     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 56)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 60, 60, 56)        28280     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 30, 30, 56)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 28, 28, 56)        28280     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 14, 14, 56)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10976)             0         
                                                                 
 dropout (Dropout)           (None, 10976)             0         
                                                                 
 dense (Dense)               (None, 128)               1405056   
                                                                 
 batch_normalization (BatchN  (None, 128)              512       
 ormalization)                                                   
                                                                 
 dense_1 (Dense)             (None, 4)                 516       
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 4)                 0         
                                                                 
=================================================================
Total params: 1,492,492
Trainable params: 1,492,236
Non-trainable params: 256
```
### Speeds, Sizes, Times

The model has 1,492,236 trainable params and 256 non-trainable params, for a total of 1,492,492 params.
It was trained on a Google Colab Pro instance for 39.9min minutes (1000 epochs).
# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

Testing data was composed of a random 10% split of the dataset.

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The metrics collected are:  `mse,rmse`
## Results

```
Mean Squared Error: 74.19
Root Mean Squared Error: 8.614
```
