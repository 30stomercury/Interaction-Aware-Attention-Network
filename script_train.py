from model import *
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('-lr', dest='lr', default=1e-4, type=float)
parser.add_argument('-keep_proba', dest='keep_proba', default=0.9, type=float)
parser.add_argument('-seq_dim', dest='seq_dim', default=256*2, type=int)
parser.add_argument('-batch_size', dest='batch_size', default=64, type=int)
args = parser.parse_args()

# cross validation
val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
# training
for val_ in val:
    print('\n###################################################validation set: {}###################################################\n\n'.format(val_))
    tf.reset_default_graph()
    # params
    hp.lr = args.lr
    hp.SEQ_DIM = args.seq_dim
    hp.batch_size = args.batch_size
    hp.keep_proba = args.keep_proba
    hp.val_set = val_
    hp.model_path_save='./model/iaan_512/{}'.format(val_)
    # generate training data/val data
    generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, val_set=hp.val_set)
    # train 
    model_ = model()
    model_.training()
