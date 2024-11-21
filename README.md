# STA-221-Project
Deep Learning Image Classification

## UC Davis MSBC cluster usage:
### Useful website: https://msbc-cluster.ucdavis.edu/howto

### 1. How to Login:
```
(your computer)$ssh username@ad3.ucdavis.edu@msbc.ucdavis.edu
```
Notes for successful Login: the ```username``` is your username for Central Authentication Service (CAS), and the keyword is your CAS keyword.

### 2. How to Open a GPU-enabled ipynb:
```
(your computer)$ ssh -L 8888:gpu-6.msbc.ucdavis.edu:8888 username@ad3.ucdavis.edu@msbc.ucdavis.edu 
(msbc)-~$ srun --gpus=1g.10gb --pty /bin/bash -i
(gpu-6)-~$ conda activate jupyter
(jupyter) (gpu-6)-~$ jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```
Note: gpu-6 above should be replaced with whichever node available as shown by sinfo (typically gpu-6) and some port between 8100-8900.

Then, open a browser on your computer and open the url provided inside the notebook server e.g. https://127.0.0.1/lab?Token=...
If everything goes fine, you are able to open a ```.ipynb``` file from your local browser and check the GPU availablility with
```
!nvidia-smi
```
If it goes wrong and further if you want to check the status of Port Forwarding ```ssh -L 8888:ip:8888``` you can use the following command *on your local terminal*:
```
netstat --tuln | grep 8888
```
If seeing the following then it means your computer is listening to any incoming external information (i.e. the problem is not in here):
```
tcp 0 0 127.0.0.1:8888 0.0.0.0:* LISTEN
```
If the problem remains:
1. Check if the ```port=8888``` is already occupied.
2. Change the notebook's ip address explicitly to ```ip=127.0.0.1```

### 3. How to transfer files:
Download:
```
scp username@ad3.ucdavis.edu@msbc.ucdavis.edu:path/to/cluster/file path/to/local/file
```
Upload:
```
scp path/to/local/file username@ad3.ucdavis.edu@msbc.ucdavis.edu:path/to/cluster/file
```

## Data Source:
### Training Dataset (with 10015 images): https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
### Testing Dataset (with 1511 images): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 

From the above website, We need to download: 	
```ISIC2018_Task3_Test_GroundTruth.tab``` and 	```ISIC2018_Task3_Test_Images.zip```

## Conda Environment on your PC Setup:
You can use ```local_cpu_env.yml``` to re-create my configured local environment with (in your Anaconda Prompt):
```
conda env create -f local_cpu_env.yml
```
After building up the new environment, activate it by:
```
conda activate DL_image # DL_image is the env name
```
## Conda environment on the HPC Cluster  
To be shared. Stay tuned.

## Reference:
https://github.com/Woodman718/FixCaps
