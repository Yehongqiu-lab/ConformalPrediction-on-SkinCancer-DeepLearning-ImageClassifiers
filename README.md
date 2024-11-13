# STA-221-Project
Deep Learning Image Classification

## Data Source:
### Training Dataset (with 10015 images): https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
### Testing Dataset (with 1511 images): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 

From the above website, We need to download: 	
```ISIC2018_Task3_Test_GroundTruth.tab``` and 	```ISIC2018_Task3_Test_Images.zip```

## Conda Environment Setup:
You can use ```environment.yaml``` to recreate my configured environment with (in your Anaconda Prompt):
```
conda env create -f environment.yaml
```
After building up the new environment, activate it by:
```
conda activate DL_image # DL_image is the env name
```
## Reference:
https://github.com/Woodman718/FixCaps
