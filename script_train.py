# supress future  warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import json
import os
import sys
import logging
from model import *
import json
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
parser.add_argument('--keep_proba', dest='keep_proba', default=0.9, type=float)
parser.add_argument('--seq_dim', dest='seq_dim', default=256, type=int)
parser.add_argument('--in_dim', dest='in_dim', default=45, type=int)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
parser.add_argument('--atten_size', dest='atten_size', default=16, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
parser.add_argument('--model_dir', dest='model_dir', default='./model/iaan/', type=str)
parser.add_argument('--feat_dir', dest='feat_dir', default='./data/feats_pooled.pkl', type=str)
parser.add_argument('--record_file', dest='record_file', default='./outputs/params.json', type=str)
args = parser.parse_args()


# cross validation
val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
results = vars(args)

# dialog order
dialog_dict = joblib.load('./data/dialog.pkl')
# feature set
seq_dict = joblib.load(args.feat_dir)
# labels
emo_all_dict = joblib.load('./data/emo_all.pkl')
# generate train/test dataframe
generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, mode='context')

# training
logging.info("Model dir:", args.model_dir)
logging.info(results)
for val_ in val:
    logging.info('\n###################################################validation set: {}###################################################\n\n'.format(val_))
    tf.reset_default_graph()

    # params
    hp.lr = args.lr
    hp.IN_DIM = args.in_dim
    hp.SEQ_DIM = args.seq_dim
    hp.ATTEN_SIZE = args.atten_size
    hp.HIDDEN_DIM = args.hidden_dim
    hp.batch_size = args.batch_size
    hp.keep_proba = args.keep_proba
    hp.val_set = val_
    hp.model_path_save = args.model_dir + val_ 

    # generate training data/val data
    generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, val_set=hp.val_set)

    # train 
    iaan = model(seq_dict)
    optimal_step = iaan.training()
    results[val_] = int(optimal_step)

json.dump(results, open(args.record_file, "w"))
