

<div align="center">    
 
# Embracing firefly flash pattern variability with data-driven species classification     
</div>
 
## Description   
Training the model uses representative sequences for each of the North American firefly species in the dataset to teach a recurrent neural network about the temporal relationships between flash events for each species. Testing the model uses timeseries generated from real videos of several species as test data to be classified. 

## How to run   
First, clone from github and create a virtual environment in which to install dependencies   
```
git clone https://github.com/peleglab/FireflyClassification
cd FireflyClassification
pip install venv
python -m venv classification_env
source ./classification_env/bin/activate 
pip install -r requirements.txt
 ```   
This would be best done on a machine that has GPU access. The code will automatically use GPUs, if they are available, and this dramatically speeds up the training process. However, they are not required to run.

Many parameters are exposed at runtime. You can change the number of samples, the depth of the network, or any of the hyperparameters, as well as name the version and specify how many classes you want to run with. Some of these are shown below, and are usefiul for training.
```
--n_layers 2 --batch_size 8 --learning_rate 0.00001  --epochs 500 --version 2 --n_classes 7 --downsample
```

If you do not specify the '--data_file' argument, the model will run with our dataset by default. However, extensions of the dataset can be made, and passed in as a different data file using that argument. If you do extend the dataset and re-train the model, please make sure the columns of the data match what we have provided in 'flash_pattern_data.csv'.

A useful test of your installation would be running some inference from the loaded checkpoints. Each of the checkpoints represents a fully realized training run with a different data split. 

```
python nn_lightning.py gru --resume_from pretrained_rnn/rnn_iter_{2.0}.ckpt
```

This will generate a report of the model results in the terminal, as well as confusion matrices and ROC / PR curves for the particular version and test set in the /figs folder.

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.lightning_rnn import LitGRU
from pytorch_lightning import Trainer

# model
model = LitGRU(params)

# train
trainer = Trainer()
trainer.fit(model, train, val)

```

## Data
The full data csv is housed in data/real_data/flash_sequence_data.csv. The units for the columns are as follows:
Dataset, species, species_label = string, no units

start_time = Hours:Minutes

start_temp_F = Degrees Fahrenheit

Num_flashes = unitless integer

Flash_duration, ifi = seconds

Timeseries = each point is a frame at 30 frames per second
