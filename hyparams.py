class hparams:
    seqlength = 855     # Maximum sequence length (sequences longer than this are dropped)
    IN_DIM = 512        # Feature dimension 
    SEQ_DIM = 128
    HIDDEN_DIM = 64
    BATCH_SIZE = 16
    ATTEN_SIZE = 16
    keep_proba = 0.9
    weight_decay = 1e-3
    num_train_steps = 3000 
    lr = 1e-4
    emo_train_file = 'emo_train.csv'
    emo_test_file = 'emo_test.csv'
    model_path_save='./model/model'
    model_path_load='./model/model'
    val_set = 'Ses01'
    

