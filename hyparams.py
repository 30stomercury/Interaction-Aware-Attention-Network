class hparams:
    seqlength = 500     # Maximum sequence length (sequences longer than this are dropped)
    IN_DIM = 45         # Feature dimension of eGemaps
    SEQ_DIM = 256
    BATCH_SIZE = 64
    ATTEN_SIZE = 16
    keep_proba = 0.9
    weight_decay = 1e-3
    num_train_steps = 100 # 15000
    context_window_size = 1
    lr = 1e-4
    emo_train_file = 'emo_train.csv'
    emo_test_file = 'emo_test.csv'
    model_path_save='./model/model'
    model_path_load='./model/model'
    val_set = 'Ses01'
    

