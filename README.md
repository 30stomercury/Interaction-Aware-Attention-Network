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
+ data.py: batch generator, each training point is a tripple of (current utt of target speaker, previous utt of target speaker, previous utt of interlocutor) and the label of current utt.
  - mode: context (refers to proposed transactional contexts)
  - mode: random (refer to baseline randIAAN)
+ model.py: main codes, BiGRU with interation-aware attention. The defined transactional contexts (previous utts)  are encoded into h_p & h_r with GRU+attention.

