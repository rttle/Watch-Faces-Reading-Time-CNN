<img width="200" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/dbbeb760-7ac3-4d53-84ce-a08071725da1">

# Watch Face Images: Reading Time CNN
This repository holds an attempt to apply deep learning on generate watch face images to read the time off the watch face, the data used is provided by Eli Schwartz on Huggingface.co: https://huggingface.co/datasets/elischwartz/synthetic-watch-faces-dataset.

## Overview
This project takes generated watch face images that is in a dataset that includes the time reflected on the watch to train a model to tell the time based on the image. This repository shows how this problem was addressed as a regression problem with two outputs, Hour and Minute. The outputs were feature engineered from the time given in the original dataset to be floats in preparation for the regression problem. Augmentation of the images were also applied. The model was trained through use of a pretrained CNN, EfficientNetB0. The best model had a MAE of 0.3799, and that model had no augmentation, standardized the target variables, and 100 epochs.

## Summary of Workdone
### Data
- Data: Type: Images
  - Input: 4 parquet files totaling to 11,000 images, output: Hour, Minute
 	- Hour = Hour + (Minutes/60)
	- Minute = Minute
- Size: 
  - watch_faces_train: 8,000 images + Time column
  - watch_faces_val: 1,000 images + Time column
  - watch_faces_test: 1,000 images + Time column
  - watch_faces_novel: 1,000 images + Time column
- Instances (Train, Validation Split): Subset of watch_faces_train used for this project.
  - Train: 800 watch face images (80%)
  - Validation: 200 watch face images (20%)
- Other optional outputs that were implemented but not used
  - Minutes from Midnight (numerical, single output)
  - Hour/Minute (categorical)
 
### Preprocessing / Clean up

**Normalization.** The scale of the numerical columns for hour and minute were widely ranged, so normalization and standardization were implemented as an options in the dataloader module.

**Image Resizing** The images were resized from (256,256) to (224,224) for use in a CNN model using EfficientNetB0 as the backbone.

### Data Visualization
The figure below shows a subset of the watch images.
<img width="257" height="209" alt="image" src="https://github.com/user-attachments/assets/c5631b62-1f20-45c8-a35f-a7e85a6774c5" />

 
The figure below shows data augmentation applied to a single image.
<img width="396" height="395" alt="image" src="https://github.com/user-attachments/assets/06558c94-1ccb-47b3-933e-c1e7034ae238" />

 
The figure below is a numerical summary of the target variables. The original column holding the time information can be seen here as a datetime type column. Of value is looking at the ranges for the different target variable columns, ignoring the datetime column of ‘text.’ This table also helps verify that the numbers used for normalization was correct.
<img width="468" height="136" alt="image" src="https://github.com/user-attachments/assets/1c97ff7c-7087-459d-8d9c-c83b494ccca1" />
 

### Problem Formulation
- Input / Output
  - Input: Preprocessed Watch Images
  - Output: Hour, Minute
- Models 
  - CNN Transferred Learning using EfficientNetB0
- Hyperparameters
  - Epochs = {25, 50, 100)
  - Batches = {64}
  - Learning Rate = {3e-4}

### Training
A base model was created for when augmentation was implemented and when augmentation was not implemented. The base models had the same specs of: standardized target variables, image size of (224,224), 1000 image dataset, 64 batches, 100 epochs, and a learning rate of 3e-4. Based on those base models, additional models were trained to see if better results could be found. These four additional models focused on changing the scaling (standardization vs. normalization) and increasing epochs. Augmentation was also readdressed despite the Augmentation Base Model performing worse than the No Augmentation Base Model.
The limited hyperparameter training was due to time constraints. A lot of troubleshooting was also done and impacted the dataloader due to working under less than ideal conditions that resulted in poor quality control of the code found throughout the notebooks. These circumstances ate a significant amount of time, and due to hardware limitations, more hyperparameter training could not be done in a timely manner. However, the issues were address enough to get a working model even if a better one could be trained given more time.

### Performance Comparison
This project approached telling the time from a watch image as a regression problem; thus, MSE and MAE were the metrics used to evaluate the models. MSE was used for the loss functions, while MAE was the main metric used for comparing the different models. Scatterplots were also used to visual the model performance by plotting Prediction vs. True values.

The figure below is a table showing the metrics of the different models. Please note that Test2 used normalization as the scaling, so the range of the output values ranged from 0-1, making the metrics look smaller (better) than the rests, but as will be shown, it was actually performing worse. Thus, the best forming model is Test4.
 <img width="468" height="80" alt="image" src="https://github.com/user-attachments/assets/673d7ec0-0d46-4ce9-b007-758a4cd33034" />


The figures below showcase the scatterplots for NoAugBase, Test2, and Test3. These plots show how Test3 actually has a tighter spread around the x=y line, while Test2 had larger spread (error). The base model is shown for reference.

**NoAugBase Model**
<img width="468" height="179" alt="image" src="https://github.com/user-attachments/assets/0f4b8147-dba2-42dd-9672-b241d3ef5a0e" />
 
**Test2**
<img width="468" height="179" alt="image" src="https://github.com/user-attachments/assets/d30062df-d278-4ccd-bf28-a5f7f9ae2a82" />

**Test3**
<img width="468" height="179" alt="image" src="https://github.com/user-attachments/assets/f76cdbe4-ec90-48d3-98fa-e7567a6f77f2" />
 


### Conclusions
Of the models trained, Test 3 did the best at telling the time when given an image of a watch face. Test 3’s model did not make use of augmentation but had 100 epochs and standardized outputs.  

### Future Work
Further hyperparameter training could help the model better read the time based on the image. Investigation into other backbones for the CNN could also lead to better results. If hardware is not a limitation, using EfficientNetB1-7 would be worth trying. ResNet is another option to explore. More preprocessing could also be done; in particular, trying other augmentations or tweaking the current augmentations. Given that the image needs to still retain a clear indication of the 12 position, the augmentations implemented in this project was very conservative. With more time, more augmentation could be explored and tested to see how that affects the model.
Instead of focusing on bettering the model, another interest could be decreasing the dataset size (1000 images used in the models produced in this repository) to see how small of a dataset could be used to create a model giving nontrivial results.

## How to reproduce results
To reproduce results, download the parquet files from the linked repository. Then ensure that the Vision_Dataloader.py file is downloaded from this repository and run the Model_Testing.ipynb notebook also found in this repository to recreate the best model (Test 3). If the desire to create all models, also run the NoAugBaseModel.ipynb and AugBaseModel.ipynb notebooks.

## Overview of files in repository
- **Vision_Preprocessing.ipynb:** Notebook that takes the dataset and prepares it for modeling. Prepares images and output variables as a tensor dataset.
- **NoAugBaseModel.ipynb:** Notebook that contains the base trained model that does not apply augmentation to the dataset. Scaling used was standardization.
- **AugBaseModel.ipynb:** Notebook that contains the base trained model that does apply augmentation to the dataset. Scaling used was standardization.
- **Model_Testing.ipynb:** Notebook that expands on the base models, resulting in stronger models. Takes the dataset, trains multiple models, compares the models through a metrics table. Different configurations were tested to find the best model, in particular what was tested was the scaling (standardization vs. normalization) and number of epochs.
- **Vision_Dataloader.py:** Module created to wrap all preprocessing done to the dataset in the feasibility notebook. Arguments for dataloader function also gives ability to change: scaling, image size, and target output.

## Data
Data is from Eli Schwartz on Huggingface.co. https://huggingface.co/datasets/elischwartz/synthetic-watch-faces-dataset

## Citations
@dataset{synthetic_watch_faces_dataset,
  author = {Eli Schwartz},
  title = {Synthetic Watch Faces Dataset},
  year = {2025},
  url = {https://huggingface.co/datasets/elischwartz/synthetic-watch-faces-dataset}
} 
<img width="468" height="636" alt="image" src="https://github.com/user-attachments/assets/d13ccd2b-f386-410d-acfb-2a04f1b54c50" />
