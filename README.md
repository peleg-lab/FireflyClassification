

<div align="center">    
 
# Embracing firefly flash pattern variability with data-driven species classification     
</div>
 
## Description   
Training the model uses representative sequences for each of the North American firefly species in the dataset to teach a recurrent neural network about the temporal relationships between flash events for each species. Testing the model uses timeseries generated from real videos of several species as test data to be classified. 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/peleglab/FireflyClassification
```
# install project   
```
cd FireFlyML   
pip install -r requirements.txt
 ```   

Many parameters are exposed at runtime. You can change the number of samples, the depth of the network, or any of the hyperparameters, as well as name the version and specify how many classes you want to run with.
```
--n_samples 1000 --n_layers 2 --batch_size 8 --learning_rate 0.00001  --epochs 100 --version 2 --n_classes 4 --downsample
```

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
