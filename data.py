import numpy as np
import pandas as pd
import random
import joblib
from random import shuffle
from glob import glob
import tensorflow as tf
from sklearn import preprocessing
from hyparams import hparams as hp

### pooling input sequence ###
def mean_pool(dic, dim=hp.IN_DIM, step=5, max_step=2500):
    new_dic = {}
    dic_ = dic
    l = list(dic_.keys())
    for l_ in l:
        if len(dic_[l_]) >= max_step:
            dic_[l_] = dic_[l_][:max_step]
        if len(dic_[l_]) % step != 0:
            for i in range(5-(len(dic_[l_]) % step)):
                dic_[l_] = np.vstack([dic_[l_], np.zeros(dim)])
        a = np.zeros([len(dic_[l_])//step, dim])
        for i in range(len(dic_[l_])//step):
            a[i] = np.mean(np.split(dic_[l_], len(dic_[l_])/step)[i], 0)
        new_dic[l_] = a
    return new_dic

### generate dialog order from transcripts ###
def generate_dialog_order():
    file = [i.replace('\\', '/') for i in glob('./Transcripts/*')]
    dialog = {}
    for f in file:
        transcripts = [j.replace('\\', '/') for j in glob(f + '/transcriptions/*')]
        for i in transcripts:
            df = pd.read_csv(i, sep=':', header=None, error_bad_lines=False)
            dialog_order = [df[0].values[i].split(' ')[0] for i in range(len(df)) if len(df[0].values[i].split(' ')[0]) >= 10 and 'XX' not in df[0].values[i].split(' ')[0]]
            dialog[i.split('/')[-1].split('.')[0]] = dialog_order
        if f == './Transcripts/Session2':
            a = glob('./lab/*')[0].replace('\\', '/')
            lab = [i.replace('\\', '/') for i in glob(a+'/*')]
            for l in lab:
                l1 = pd.read_csv(l)
                l2 = pd.read_csv(l.replace('Ses02_F', 'Ses02_M'))
                dialog_order = []
                for i in range(len(l1)):
                    dialog_order.append((float(l1.values[i][0].split(' ')[0]), l1.values[i][0].split(' ')[-1]))
                for i in range(len(l2)):
                    dialog_order.append((float(l2.values[i][0].split(' ')[0]), l2.values[i][0].split(' ')[-1])) 
                dialog_order = sorted(dialog_order)
                dialog_order = [i[-1] for i in sorted(dialog_order)]
                dialog[l.split('/')[-1].split('.')[0]] = dialog_order
    return dialog

### generate interaction training pairs ###
# total 4 class/ total 5531 emo labels/ total 94 emo label dropped due to lack of previous sentence
def generate_interaction_sample(index_words, seq_dict, emo_dict, seqlength=hp.seqlength):
    emo = ['ang', 'hap', 'neu', 'sad']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    for index, center in enumerate(index_words):
        if emo_dict[center] in emo:
            center_.append(center)
            center_label.append(emo_dict[center])
            pt = []
            pp = []
            for word in index_words[max(0, index - 8): index]:
                if word[-4] == center[-4]:
                    pt.append(word)
                else:
                    pp.append(word)

            if len(pt) != 0:
                target_.append(pt[-1])
                target_label.append(emo_dict[pt[-1]])
                target_dist.append(index - index_words.index(pt[-1]))
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist

def generate_random_sample(index_words, seq_dict, emo_dict, seqlength=hp.seqlength):
    emo = ['ang', 'hap', 'neu', 'sad']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    for index, center in enumerate(index_words):
        if emo_dict[center] in emo:
            center_.append(center)
            center_label.append(emo_dict[center])
            
            idx = [i for i in range(len(index_words)) if i != index_words.index(center)]
            shuffle(idx)
            tar_idx = idx[0]
            oppo_idx = idx[1]
            target_.append(index_words[tar_idx])
            target_label.append(emo_dict[index_words[tar_idx]])
            target_dist.append('None')

            opposite_.append(index_words[oppo_idx])
            opposite_label.append(emo_dict[index_words[oppo_idx]])
            opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist


def generate_interaction_data(dialog_dict, seq_dict, emo_dict, val_set=hp.val_set, mode='context'):
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train = [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val = [], [], [], [], [], [], [], []
    if mode=='context':
        generator = generate_interaction_sample
    elif mode == 'random':
        generator = generate_random_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        # training set
        if val_set not in k:
            c, t, o, cl, tl, ol, td, od = generator(dialog_order, seq_dict, emo_dict)
            center_train += c
            target_train += t
            opposite_train += o
            center_label_train += cl
            target_label_train += tl
            opposite_label_train += ol
            target_dist_train += td
            opposite_dist_train += od
        # validation set
        else:
            c, t, o, cl, tl, ol, td, od = generator(dialog_order, seq_dict, emo_dict)
            center_val += c
            target_val += t
            opposite_val += o
            center_label_val += cl
            target_label_val += tl
            opposite_label_val += ol
            target_dist_val += td
            opposite_dist_val += od

    # save dialog pairs to train.csv and test.csv
    train_filename= hp.emo_train_file
    val_filename= hp.emo_test_file
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)
    # validation
    d = {'center': center_val, 'target': target_val, 'opposite': opposite_val, 'center_label': center_label_val, 
         'target_label': target_label_val, 'opposite_label': opposite_label_val, 'target_dist': target_dist_val, 'opposite_dist': opposite_dist_val}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(val_filename, sep=',', index = False)

class interaction_data_generator:
  def __init__(self, batch_size, seqlength, seq_dict, file_path, latency=1.0, mode=None):
    df = pd.read_csv(file_path)
    self.center = df['center']
    self.target = df['target']
    self.opposite = df['opposite']
    le = preprocessing.LabelEncoder()
    self.label = le.fit_transform(df['center_label'])
    self.batch_size = batch_size
    self.seqlength = seqlength
    self.seq_dict = seq_dict
    self.latency = latency
    # mode
    self.mode = mode
    lst = [i for i in range(len(self.center))]
    if self.mode == 'bucketing':
        # sequence length for bucketing
        self.lst = sorted(lst, key=lambda i: max(len(self.seq_dict[self.center[i]]), len(self.seq_dict[self.target[i]]), len(self.seq_dict[self.opposite[i]])))# sort the data by length
        self.seq_len = sorted([max(len(self.seq_dict[self.center[i]]), len(self.seq_dict[self.target[i]]), len(self.seq_dict[self.opposite[i]])) for i in range(len(self.center))])
    else:
        shuffle(lst)
        self.lst = lst
        self.seq_len = [max(len(self.seq_dict[self.center[i]]), len(self.seq_dict[self.target[i]]), len(self.seq_dict[self.opposite[i]])) for i in self.lst]
    # initialize data generator
    self.single_gen = self.generate_sample()


  def generate_sample(self):
    for l, sl in zip(self.lst, self.seq_len):
        c_ = self.seq_dict[self.center[l]]
        p_ = self.seq_dict[self.target[l]]
        o_ = self.seq_dict[self.opposite[l]]
        l_ = self.label[l]
        if self.latency != 1.0:
            c_ = c_[:int(self.latency*len(c_))]

        xs= np.zeros([self.seqlength, hp.IN_DIM])
        ps = np.zeros([self.seqlength, hp.IN_DIM])                                                                                                                                                                 
        os = np.zeros([self.seqlength, hp.IN_DIM])
        xs[:len(c_)] = c_
        ps[:len(p_)] = p_
        os[:len(o_)] = o_

        if self.mode != 'bucketing' and sl > 0:
            sl_ = sl

        yield xs, ps, os, l_, sl
   
  def get_batch(self):
    while True:
        center_batch = np.zeros([self.batch_size, self.seqlength, hp.IN_DIM], dtype=np.float32)
        target_batch = np.zeros([self.batch_size, self.seqlength, hp.IN_DIM], dtype=np.float32)
        opposite_batch = np.zeros([self.batch_size, self.seqlength, hp.IN_DIM], dtype=np.float32)
        y_batch = np.zeros([self.batch_size])
        for index in range(self.batch_size):
            center_batch[index], target_batch[index], opposite_batch[index], y_batch[index], sl = next(self.single_gen)
                
        yield center_batch[:, :sl, :], target_batch[:, :sl, :], opposite_batch[:, :sl, :], y_batch

  def get_test_batch(self):
    num_test = len(self.center)
    num_test_steps = num_test // self.batch_size + 1
    last_batch = num_test % self.batch_size
    for step in range(num_test_steps):
        if step != num_test_steps -1:
            bs = self.batch_size
        else:
            bs = last_batch
        center_batch = np.zeros([bs, self.seqlength, hp.IN_DIM], dtype=np.float32)
        target_batch = np.zeros([bs, self.seqlength, hp.IN_DIM], dtype=np.float32)
        opposite_batch = np.zeros([bs, self.seqlength, hp.IN_DIM], dtype=np.float32)
        y_batch = np.zeros([bs])
        for index in range(bs):
            center_batch[index], target_batch[index], opposite_batch[index], y_batch[index], sl = next(self.single_gen)

        yield center_batch[:, :sl, :], target_batch[:, :sl, :], opposite_batch[:, :sl, :], y_batch


#if __name__ == "__main__":
# dialog order
dialog_dict = joblib.load('./data/dialog.pkl')
# feature set
seq_dict = joblib.load('./data/feat_pooled.pkl')
# labels
emo_all_dict = joblib.load('./data/emo_all.pkl')
# generate train/test dataframe
generate_interaction_data(dialog_dict, seq_dict, emo_all_dict, mode='context')
