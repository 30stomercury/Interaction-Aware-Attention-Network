from model import *
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('-seq_dim', dest='seq_dim', default=256*2, type=int)
args = parser.parse_args()

# cross validation
val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
# selected ckpt
ckpt = [1000, 700, 1000, 1000, 1000]
hp.SEQ_DIM=args.seq_dim; hp.ATTEN_SIZE=16
gt, pred = [], []
t_effect, o_effect, b = [], [], []
for val_, ckpt_ in zip(val, ckpt):
    tf.reset_default_graph()
    # params
    hp.val_set = val_
    hp.model_path_load='./model/iaan/'+val_
    hp.keep_proba=1
    print("################{}################".format(val_))
    # generate training data/val data
    generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, val_set=hp.val_set)
    # testing
    model_ = model()
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    model_.restore(ckpt_)
    model.is_training = False

    test_gt, p = model_.testing()
    pred += p
    gt += test_gt
    print(test_gt)
    
ave_uar = recall_score(gt, pred, average='macro')
print('ave uar over all data: {}'.format(ave_uar))
