from model import *

# cross validation
val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
# training
for val_ in val:
    print('\n###################################################validation set: {}###################################################\n\n'.format(val_))
    tf.reset_default_graph()
    # params
    hp.val_set = val_
    hp.model_path_save='./model/iaan/{}'.format(val_)
    # generate training data/val data
    generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, val_set=hp.val_set)
    # train 
    model_ = model()
    model_.training()
