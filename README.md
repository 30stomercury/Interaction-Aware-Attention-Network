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
    1. interaction_data_generator(): generate data (emo_train.csv & emo_test.csv) under specific modes.
        - mode: context (referred to proposed transactional contexts)
        - mode: random (refered to baseline randIAAN)
    2. interaction_data_generator(): batch generator.

+ **model.py:** 
main codes.

+ **hyparams.py:** hyperparameters.

+ **script_train.py/script_test.py:** training/testing script. To evaluate under realistic scenearios of our model, we adopt leave-onesession-out cross validation.

## Docker image
Coming soon.
