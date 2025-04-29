
# Highly Undersampled MRI Reconstruction via a Single Posterior Sampling of Diffusion Models
- This repository contains the implementation of our SSDM-MRI method.
- Both Complex_Data and Real_Data folders follow a similar reconstruction pipeline, with slight code differences tailored to the data type: Complex_Data supports both magnitude and phase reconstruction, while Real_Data is for magnitude-only reconstruction.

## Requirements
  bash requirements.sh

## Training
### If you want to train your own model from scratch, take Real_Data as an example:
####(1)enter the Real_Data directory, modify the train path and other training parameters in the config/img_restoration.json file,
#####"datasets": { // train or test
   #####     "train": {
       #####     "which_dataset": {  // import designated dataset using arguments
        #####        "name": ["data.dataset", "MRI_Restoration"], // import Dataset() class / function(not recommend) from dataset.dataset.py (default is [dataset.dataset.py])
           #####     "args":{ // arguments to initialize dataset
            #####        "data_root": "train",
           #####         "acc_factor": -1
        #####        }
    #####        }
### (2)then run the following command:  
#### python run.py -p train -c config/img_restoration.json



## Sampling 
### 

