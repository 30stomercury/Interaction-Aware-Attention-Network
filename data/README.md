# Data Overview

## Folder structure
```
data/
└── wav/
    ├── S1
    │   └── wav_mod
    ├── S2
    │   └── wav_mod
    ├── S3
    │   └── wav_mod
    ├── S4
    │   └── wav_mod
    └── S5
        └── wav_mod
```

## Feature extraction pipline
```
# pull image from dockerhub
docker pull ff936tw/iaan:v2

# run container
docker run --runtime=nvidia -v /path/to/wav/folder: \ 
/workspace/Interaction-aware_Attention_Network/data/wav -it ff936tw/iaan:v2

# extracted audio feature by opensmile toolkit
python3 extractSMILE.py -audio_folder wav/ -features_folder extracted_features/

# Select 45-dimensional features and do speaker normalization. 
python3 preprocess.py -features_folder extracted_features/

You can find the original dataset and extracted feature:
https://drive.google.com/drive/u/1/folders/1rJk4V5YeQNTOtT0WjZ-VlEzWg4c6-q1R

```

## Descriptions
+ feat_all.pkl: 
    - key: speaker ID
    - value: 45 dimensional features extracted by openSMILE.
    
+ feat_pooled.pkl: (feature pooled every n frames)
    - You can run function mean_pool() in data.py, in this paper we set n=5.
    - key: speaker ID
    - value: 45 dimensional features extracted by openSMILE.

+ dialog.pkl:
    - key: dialog ID
    - value: dialog order of speaker ID

+ emo_all.pkl:
    - key: speaker ID
    - value: emotion labels

