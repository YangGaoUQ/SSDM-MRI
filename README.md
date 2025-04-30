
# Highly Undersampled MRI Reconstruction via a Single Posterior Sampling of Diffusion Models
- This repository contains the implementation of our SSDM-MRI method.
- Both Complex_Data and Real_Data folders follow a similar reconstruction pipeline, with slight code differences tailored to the data type: Complex_Data supports both magnitude and phase reconstruction, while Real_Data is for magnitude-only reconstruction.

## Requirements
```python
bash requirements.sh
```


## Pre-Training and Distillation
If you want to train your own model from scratch, take Real_Data as an example:

(1)Enter the Real_Data directory, modify the train path and other training parameters in the config/img_restoration.json file:

```yaml
    "datasets": { // train or test
        "train": {
            "which_dataset": {  // import designated dataset using arguments
                "name": ["data.dataset", "MRI_Restoration"], // import Dataset() class / function(not recommend) from dataset.dataset.py (default is [dataset.dataset.py])
                "args":{ // arguments to initialize dataset
                    "data_root": "train",
                    "acc_factor": -1,
                    "mask_type": "gaussian1d"
                }
            },
```

(2)Then run the following command:  
```python
python run.py -p train -c config/img_restoration.json
```

(3)After completing the pre-training of the model, you can run distillate.py for distillation

## Sampling 
If you want to test with a pre-trained model, still using Real_Data as an example:

(1) Download the corresponding pre-trained model here [Google Drive](). Create the "checkpoints" folder and put the pre-trained model in it.

(2) Modify the following entries in the config/img_restoration.json file:
```yaml
        "test": {
            "which_dataset": {
                "name": ["data.dataset","MRI_Restoration"], // import Dataset() class / function(not recommend) from default file
                "args":{
                    "data_root": "demo",
                    "acc_factor":8,
                    "mask_type": "gaussian1d"
                }
            },
```

(3) Then run the following command:  
```python
python run.py -p test -c config/img_restoration.json
```

