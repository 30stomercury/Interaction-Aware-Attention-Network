# Interaction-aware Attention Network
+ A tensorflow implementation of Interaction-aware Attention Network in [An Interaction-aware Attention Network for Speech Emotion Recognition in Spoken Dialogs](https://ieeexplore.ieee.org/document/8683293/references#references).

## Data:
Data disciptions of IEMOCAP please refer to [here](https://sail.usc.edu/iemocap/).

## Requirements
Some required libraries:
```
python                   >=3.6   
tensorflow-gpu           1.11.0
joblib   		 0.13.0
pandas                   0.22.0
scikit-learn             0.19.1
numpy			 1.15.3
```
## Code:

|  codes   |  descriptions |
|:--------:|:-------------:|
| data.py  |  Includes batch generator & data generator.  |
| model.py |  main codes.  |
| hyparams.py |hyperparameters|
| script_train.py |testing scripts|
| script_test.py |training scripts|

To evaluate under realistic scenarios of our model, we adopt leave-one-session-out cross validation.

## Run:
For training:  
```
python3 script_train.py -lr 0.0001 \  
                         -batch_size 64 \ 
                         -keep_proba 0.9 \ 
                         -seq_dim 512 \ 
                         -save_path ./model/iaan/ > out.txt
```
For testing, please specify the checkpoint (ckpt) for each session in line 12:  
```
python3 script_test.py -seq_dim 512
```

## Docker image
sudo docker pull ff936tw/iaan:v2
