
The main folders composing the project are:
 - The src/notebook folder: contains the final notebook of the project and other utils written during the project but no more used.
   The python notebook clones this repository and extensiely use its code

 - The src/main folder contains the core of this project. It contains the models that have been trained, alongside with the relevant code to use them and postprocess the results.
 
 - The src/telegrambot folder contains the code used to spin up the telegram bot. This bot can receive the picture of a car and answer with the predicted plate(s). 

Extra:

The src/preprocessing folder contains all of the code written to merge the different datasets, used at the beginning of this project.
It is used, among the other things, to resize the images and consequently their annotations.
Most of it is useless in the final version of the project, because those datasets have been discarded.

The datasets folder contains some utils to deal with the dataset, such as loading the dataset and showing the images and the bounding boxes.
 
The assets folder contains assets used for the report of the project.