from model import *
import os
import json
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('--seq_dim', dest='seq_dim', default=256, type=int)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
parser.add_argument('--in_dim', dest='in_dim', default=45, type=int)
parser.add_argument('--atten_size', dest='atten_size', default=16, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
parser.add_argument('--num_steps', dest='num_steps', default=2000, type=int)
parser.add_argument('--model_dir', dest='model_dir', default='./model/iaan/', type=str)
parser.add_argument('--feat_dir', dest='feat_dir', default='./data/feats.pkl', type=str)
parser.add_argument('--out_dir', dest='out_dir', default='./outputs/', type=str)
parser.add_argument('--result_file', dest='result_file', default='output/iaan.json', type=str)
args = parser.parse_args()

# cross validation
val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
filepath = args.result_file

if not os.path.isfile(filepath):
    ckpt = [args.num_steps]*5
    print("Parameters:", vars(args))
    hp.SEQ_DIM = args.seq_dim
    hp.HIDDEN_DIM = args.hidden_dim
    hp.ATTEN_SIZE = args.atten_size
    hp.batch_size = args.batch_size
    
else:
    with open(filepath, 'r') as f:
        results = json.load(f)
    print("Parameters:", results)
    # selected ckpt
    ckpt = [results[val_] for val_ in val] 
    hp.SEQ_DIM = results['seq_dim']
    hp.IN_DIM = results['in_dim']
    hp.ATTEN_SIZE = results['atten_size']
    hp.batch_size = args.batch_size
    args.model_dir = results['model_dir']
    args.feat_dir = results['feat_dir']
    hp.HIDDEN_DIM = results['hidden_dim']

# dialog order
dialog_dict = joblib.load('./data/dialog.pkl')
# feature set
seq_dict = joblib.load(args.feat_dir)
# label
emo_all_dict = joblib.load('./data/emo_all.pkl')
# generate train/test dataframe
generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, mode='context')

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
gt, pred = [], []
t_effect, o_effect, b = [], [], []
for val_, ckpt_ in zip(val, ckpt):
    tf.reset_default_graph()
    # params
    hp.val_set = val_
    hp.keep_proba=1
    hp.model_path_load = args.model_dir + val_ 
    print("################{}################".format(val_))
    # generate training data/val data
    generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, val_set=hp.val_set)
    # testing
    iaan = model(seq_dict)
    iaan.is_training = False
    iaan.restore(ckpt_)

    test_gt, p, _, _ = iaan.testing()
    pred += p
    gt += test_gt

ave_uar, ave_acc = evaluation(gt, pred)
results['uar'] = ave_uar
results['war'] = ave_acc
json.dump(results, open(args.result_file.split('.')[0]+'_results.json', "w"))
