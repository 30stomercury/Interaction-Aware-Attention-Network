import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from hyparams import hparams as hp
from ops import *
from data import *
import os


class model:
  def __init__(self,
               is_training=True):

    self.is_training = is_training
    self._get_session()  # get session
    self._build_model()
    self._get_emo_iter()
    

  def _build_model(self):

    with tf.variable_scope('intput'):
      # current utt: Uc
      self.current_utt = tf.placeholder(dtype=tf.float32, shape=[None, None, hp.IN_DIM])
      # previous utt of target speaker: Up
      self.target_utt = tf.placeholder(dtype=tf.float32, shape=[None, None, hp.IN_DIM])
      # previous utt of opposite speaker: Ur
      self.opposite_utt = tf.placeholder(dtype=tf.float32, shape=[None, None, hp.IN_DIM])
      # gt
      self.groundtruths = tf.placeholder(dtype=tf.int64, shape=[None])

    with tf.variable_scope('UpEncoder', reuse=tf.AUTO_REUSE):
      cell = tf.contrib.rnn.GRUCell(hp.SEQ_DIM*2)
      outputs, _ = tf.nn.dynamic_rnn(cell, 
                                     inputs=self.target_utt, 
                                     dtype=tf.float32) 
      self.h_p = self_attention(outputs)

    with tf.variable_scope('UrEncoder', reuse=tf.AUTO_REUSE):
      cell = tf.contrib.rnn.GRUCell(hp.SEQ_DIM*2)
      outputs, _ = tf.nn.dynamic_rnn(cell, 
                                     inputs=self.opposite_utt, 
                                     dtype=tf.float32) 
      self.h_r = self_attention(outputs)

    with tf.variable_scope('UcEncoder', reuse=tf.AUTO_REUSE):
      ### RNN encoder ###
      self.cell_units = hp.SEQ_DIM
      # Forward cell
      self.fw_cell = tf.contrib.rnn.GRUCell(self.cell_units)
      # Backward cell
      self.bw_cell = tf.contrib.rnn.GRUCell(self.cell_units)
      # add dropout
      if self.is_training:
          self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell,
                                                      input_keep_prob=hp.keep_proba,
                                                      output_keep_prob=hp.keep_proba)
          self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell,
                                                       input_keep_prob=hp.keep_proba,
                                                      output_keep_prob=hp.keep_proba)
      else:
          self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell,
                                                      input_keep_prob=1,
                                                      output_keep_prob=1)
          self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell,
                                                       input_keep_prob=1,
                                                      output_keep_prob=1)

      #initial_state = LSTMCell.zero_state(self.hp.BATCH_SIZ, dtype=tf.float32)
      outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                             cell_bw=self.bw_cell,
                                                             inputs= self.current_utt,
                                                             dtype=tf.float32,
                                                             time_major=False)
      # hidden units of center utt
      self.c_out = tf.concat(outputs, 2)
      # add masks
      mask = mask_seq(self.current_utt)
      seq_len = tf.shape(self.c_out)[1]
      h_p_ = tf.tile(tf.expand_dims(self.h_p, 1), [1, seq_len, 1])
      h_r_ = tf.tile(tf.expand_dims(self.h_r, 1), [1, seq_len, 1])
      h_p_ = h_p_*mask
      h_r_ = h_r_*mask
      self.out = self.c_out*mask

      with tf.variable_scope('interaction-aware_attention', reuse=tf.AUTO_REUSE): 
        ### context_vec c(a, h) w/ Bahdanau attention###
        hidden_size = self.out.shape[-1].value  # hidden size of the RNN layer
        attention_size = hp.ATTEN_SIZE
        # Trainable parameters
        W_c = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W_p = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W_r = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.nn.tanh(tf.tensordot(self.out, W_c, axes=1) + tf.tensordot(h_p_, W_p, axes=1) + tf.tensordot(h_r_, W_r, axes=1) + b)
        vu = tf.tensordot(v, u, axes=1)
        # mask attention weights
        mask_att = tf.sign(tf.abs(tf.reduce_sum(self.current_utt, axis=-1))) # [batch_size, sequence_len]
        paddings = tf.ones_like(mask_att)*(-10e8)
        vu = tf.where(tf.equal(mask_att, 0), paddings, vu)               # [batch_size, sequence_len]       
        alphas = tf.nn.softmax(vu)                                       # [batch_size, sequence_len]
        # Output reduced with context vector: [batch_size, sequence_len]
        self.h_c = tf.reduce_sum(self.out * tf.expand_dims(alphas, -1), 1)


    with tf.variable_scope('MLP_emo', reuse=tf.AUTO_REUSE):
      self.out = tf.concat([self.h_c, self.h_p, self.h_r], 1)
      # fully layer 1
      out_weight1 = tf.get_variable('out_weight1', shape=[hp.SEQ_DIM*6, 64], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
      out_bias1 = tf.get_variable('out_bias1', shape=[64], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.1))
      out_weight2 = tf.get_variable('out_weight2', shape=[64, 4], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
      out_bias2 = tf.get_variable('out_bias2', shape=[4], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.1))
      dense = tf.matmul(self.out, out_weight1) + out_bias1
      dense = layer_norm(dense, hp.keep_proba, self.is_training)
      dense = tf.nn.relu(dense)
      dense = tf.matmul(dense, out_weight2) + out_bias2

      self.logits_emo = dense

    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
      # label smoothing
      self.gt_emo = tf.one_hot(self.groundtruths, depth=4)
      self.gt_emo = label_smoothing(self.gt_emo)
      # classification loss
      self.emo_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.gt_emo, logits=self.logits_emo)
      # total loss
      self.e_loss = self.emo_loss + hp.weight_decay*(tf.nn.l2_loss(out_weight1) + tf.nn.l2_loss(out_bias1) + tf.nn.l2_loss(out_weight2) + tf.nn.l2_loss(out_bias2))
      self.e_optimizer = tf.train.AdamOptimizer(hp.lr).minimize(self.e_loss)

    with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
      self.e_prediction = tf.argmax(self.logits_emo, axis=1)
      self.e_accuracy = tf.contrib.metrics.accuracy(
        labels=tf.argmax(self.gt_emo, axis=1),
        predictions=self.e_prediction)

    # Initialzation
    self.saver = tf.train.Saver(max_to_keep=2000)
    self.sess.run(tf.global_variables_initializer())

  def training(self):
    total_loss = 0
    total_acc = 0
    total_uar = 0
    Epoch = 1
    uar_list = []
    # start training
    for index in range(hp.num_train_steps):
      if index == 0:
        print('=========training emotion classification !=========')

      try:
        current_utt, target_utt, opposite_utt, groundtruths = next(self.e_train_gen)
      except StopIteration: 
	# generator has nothing left to generate
        # initialize iterator again
        print('=========Epoch {} finished !========='.format(Epoch)) 
        Epoch += 1
        self._get_emo_iter()
        current_utt, target_utt, opposite_utt, groundtruths = next(self.e_train_gen)

      fd = {
            self.current_utt: current_utt,
            self.target_utt: target_utt,
            self.opposite_utt: opposite_utt,
            self.groundtruths: groundtruths
           }

      # uar
      pred_batch = self.sess.run(self.e_prediction, feed_dict=fd)
      uar_batch = recall_score(groundtruths, pred_batch, average='macro')
      # loss & acc
      loss_batch, _, acc_batch = self.sess.run([self.e_loss, self.e_optimizer, self.e_accuracy], feed_dict=fd)  
      total_loss += loss_batch
      total_acc += acc_batch
      total_uar += uar_batch

      if (index + 1) % 20 == 0:
        print('step: {}, Ave emo loss : {:.3f}, Ave emo train acc: {:.3f}, Ave emo train uar: {:.3f}'.format(index+1,
                                                                                                              total_loss/20,
                                                                                                              total_acc/20, 
                                                                                                              total_uar/20,))
        total_loss = 0.0
        total_acc = 0.0
        total_uar = 0.0
        self.save(index)

      if (index + 1) % 100 == 0:
        test_gt, test_pred  = self.testing()
        uar_list.append(float(recall_score(test_gt, test_pred, average='macro')))

    print('optimal step: {}, optimal uar: {}'.format((np.argmax(uar_list)+1)*100, max(uar_list)))

  def testing(self):
    self.is_training = False
    keep_proba = hp.keep_proba
    hp.keep_proba = 1
    # test data length

    df = pd.read_csv(hp.emo_test_file)
    self._get_emo_iter()
    test_gen = self.e_test_gen

    num_test_steps = len(df) // (2*hp.BATCH_SIZE) + 1
    test_pred = []
    test_gt = []

    for i in range(num_test_steps):
      current_utt, target_utt, opposite_utt, groundtruths = next(test_gen)

      fd = {
            self.current_utt: current_utt,
            self.target_utt: target_utt,
            self.opposite_utt: opposite_utt,
            self.groundtruths: groundtruths
           }
      acc_batch, pred_batch = self.sess.run([self.e_accuracy, self.e_prediction], feed_dict=fd)
      uar_batch = recall_score(groundtruths, pred_batch, average='macro')
      test_pred += list(pred_batch)
      test_gt += list(groundtruths)

    print(confusion_matrix(test_gt, test_pred))
    ave_uar = recall_score(test_gt, test_pred, average='macro')
    ave_acc = accuracy_score(test_gt, test_pred)
    self.is_training = True
    hp.keep_proba = keep_proba
    print('Ave test acc: {:.3f}, Ave test uar: {:.3f}'.format(ave_acc, ave_uar))   
    return test_gt, test_pred


  def _get_emo_iter(self):
    # initialize iterator
    if self.is_training:
      e_dat_train = interaction_data_generator(hp.BATCH_SIZE, hp.seqlength, seq_dict, hp.emo_train_file)
      self.e_train_gen = e_dat_train.get_batch()
    else:
      e_dat_test = interaction_data_generator(hp.BATCH_SIZE*2, hp.seqlength, seq_dict, hp.emo_test_file, mode='bucketing')
      #e_dat_test = interaction_data_generator(hp.BATCH_SIZE*2, hp.seqlength, seq_dict, hp.emo_test_file)
      self.e_test_gen = e_dat_test.get_test_batch()

  def _get_session(self):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)   
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))    
    
  def save(self, e):
    if not os.path.exists(hp.model_path_save):
      os.makedirs(hp.model_path_save)
    self.saver.save(self.sess, hp.model_path_save+'/model_%d.ckpt' % (e + 1))

  def restore(self, e):
    self.saver.restore(self.sess, hp.model_path_load+'/model_%d.ckpt' % (e))
