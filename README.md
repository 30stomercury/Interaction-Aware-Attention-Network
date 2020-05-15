# Interaction-aware Attention Network
+ A tensorflow implementation of Interaction-aware Attention Network in [An Interaction-aware Attention Network for Speech Emotion Recognition in Spoken Dialogs](https://ieeexplore.ieee.org/document/8683293/references#references).

## Model Overview
<img width="934" alt="Screen Shot 2019-10-21 at 5 11 24 PM" src="https://user-images.githubusercontent.com/14361791/67192274-de2d7880-f425-11e9-9bf5-d85b62cfd621.png">

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
For feature pooling to reduce computational cost:
```
python3 pool_feats.py --input_file INPUT_FILE --output_file OUTPUT_FILE --feat_dim FEAT_DIM --step STEP --max_size MAX_SIZE
```

For training:  
```
python3 script_train.py --seq_dim SEQ_DIM \
                        --atten_size ATTEN_SIZE \
                        --batch_size BATCH_SIZE \
                        --model_dir MODEL_DIR \
                        --record_file outputs/RECORD_FILE.json \
                        --feat_dir data/XXX.pkl
```
For testing
```
python3 script_test.py  --result_file outputs/RECORD_FILE.json --feat_dir data/XXX.pkl
```
