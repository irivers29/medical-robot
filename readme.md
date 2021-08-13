#### Identifying the landing position of the image : Rheumatoid Arthritis application

### Steps to train U-net

1. save all images and labels in two separate folders (image and corresponding label should have the same name, i.e.: 'img00001.png')
2. define in src/data/segmentation_dataset.py the path where the images and labels are stored
3. create the .txt files in get_img_names.ipynb. For this, define the path of the images in this file
4. open the autoencoder.ipynb folder and define all necesary paths

### Dataset
Characteristics and definition - what the dataset has and what it does not have.
   1. Preprocessing dataset: Data augmentation, also understanding more on hand to understand results, edge cases
      1. Expanding dataset (via data augmentation techniques with skin color)
   2. Discuss processing architecture : 
   3. Train the model - divide training, validation etc. 
   4. Fine tune the parameters and iterate 

---
   Define knowledge into the system.
   Also we need to come up with 
   Get an image -> 
   Is the image good -> 
   Preprocess the image for inputs to the algorithm -> 
   __Process with learned model__ -> 
   __Compute landing position__ -> 
   __Check if the answer is correct__