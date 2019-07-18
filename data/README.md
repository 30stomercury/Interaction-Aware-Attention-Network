# Data Overview

## Directory structure
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

# Feature extraction pipline
```
# extracted audio feature by opensmile toolkit
python3 extractSMILE.py -audio_folder wav/ -features_folder extracted_features/

# Select 45-dimensional features and do speaker normalization. 
python3 preprocess.py -features_folder extracted_features/

```

# Descriptions
+ feat_all.pkl: (the config file and processing codes are coming soon.)
    - key: speaker ID
    - value: 45 dimensional features extracted by openSMILE.
    
+ feat_pooled.pkl: (feature pooled every 5 frames)
    - Run mean_pool() in data.py
    - key: speaker ID
    - value: 45 dimensional features extracted by openSMILE.

+ dialog.pkl:
    - key: dialog ID
    - value: dialog order of speaker ID

+ emo_all.pkl:
    - key: speaker ID
    - value: emotion labels

