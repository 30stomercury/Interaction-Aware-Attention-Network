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
+ **data.py:** 
Includes batch generator & data generator, each training point is a tripple of (current utt of target speaker, previous utt of target speaker, previous utt of interlocutor) and the label of current utt.
    1. generate_interaction_data(): generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
        - `context`: proposed transactional contexts, referred to IAAN.
        - `random`: randomly sampled contexts, referred to baseline randIAAN.
    2. interaction_data_generator(): batch generator.

+ **model.py:** 
main codes.

+ **hyparams.py:** hyperparameters.

+ **script_train.py / script_test.py:** training / testing script. To evaluate under realistic scenarios of our model, we adopt leave-one-session-out cross validation.

## Run:
+ `python3 script_train.py` for training.
+ `python3 script_test.py` for testing, please specify the checkpoint (ckpt) for each session in line 6.

## Docker image
Coming soon.
