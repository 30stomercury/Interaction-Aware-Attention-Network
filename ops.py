import tensorflow as tf
import numpy as np
from hyparams import hparams as hp
from ops import *
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

def additive_attention(inputs1, inputs2, attention_size=hp.ATTEN_SIZE, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("additive_attention", reuse=reuse):
        # hidden size of the RNN layer
        hidden_size = int(inputs1.shape[-1])  
        # Trainable parameters
        W1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W2 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.nn.tanh(tf.tensordot(inputs1, W1, axes=1) + tf.tensordot(inputs2, W2, axes=1) + b)
        vu = tf.tensordot(v, u, axes=1)   # (Batch size,T)
        alphas = tf.nn.softmax(vu)        # (Batch size,T)

        # Output reduced with context vector: (Batch size, hidden_size)
        outputs = tf.reduce_sum(inputs1 * tf.expand_dims(alphas, -1), 1)
        outputs = tf.reshape(outputs, [-1, hidden_size])
        return outputs

def self_attention(inputs, attention_size=hp.ATTEN_SIZE):
    # hidden size of the RNN layer
    hidden_size = int(inputs.shape[-1])  
    # Trainable parameters
    W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.nn.tanh(tf.tensordot(inputs, W, axes=1) + b)
    vu = tf.tensordot(v, u, axes=1)   # (Batch size,T)
    alphas = tf.nn.softmax(vu)        # (Batch size,T)
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    outputs = tf.reshape(outputs, [-1, hidden_size])

    return outputs

def scaled_dot_product_attention(queries, keys, values, 
                        keep_proba=1, is_training=True, reuse=None):
    '''
    Args:
      queries: A 3d tensor with shape of [batch_size, sequence_len_q, d_q].
      keys: A 3d tensor with shape of [batch_size, sequence_len_k, d_k].
      values: A 3d tensor with shape of [batch_size, sequence_len_k, d_v].
      is_training: dropout in training phase, closed otherwise.
      
    Returns:
      [batch_size, sequence_len, d_model] 
    '''
    with tf.variable_scope("scaled_dot_prodoct", reuse=reuse):
        # d_q = d_k
        input_size_k = keys.shape[-1]
        input_size_q = input_size_k
        input_size_v = values.shape[-1]
        # input_size_q = d_model
        Q = tf.layers.dense(queries, input_size_q, use_bias=True) # [batch_size, sequence_len_q, d_model].
        K = tf.layers.dense(keys, input_size_k, use_bias=True) # [batch_size, sequence_len_k, d_model].
        V = tf.layers.dense(keys, input_size_k, use_bias=True) # [batch_size, sequence_len_k, d_model].     
        # MatMul Q & K
        # [batch_size, sequence_len_q, d_model] x [batch_size, d_model, sequence_len_k] -> [batch_size, sequence_len_q, sequence_len_k]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) # [batch_size, sequence_len_q, sequence_len_k]        
        # Scale
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5) # devided by sqrt(d_k)
        # mask keys
        key_masks = masking(keys)
        paddings = tf.ones_like(outputs)*(-10e8)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)# [batch_size, sequence_len_q, sequence_len_k]        
        # Activation
        outputs = tf.nn.softmax(outputs) # [batch_size*h, sequence_len_q, sequence_len_k]          
        # mask queries
        queries_masks = masking(queries)
        outputs = queries_masks*outputs # [batch_size*h, sequence_len_q, sequence_len_k]        

        # Dropouts
        if is_training:
            outputs = tf.nn.dropout(outputs, keep_prob=keep_proba)               
        # MatMul V
        outputs = tf.matmul(outputs, V) # [batch_size*h, sequence_len_q, d_model/h]      
        # linear
        outputs = tf.layers.dense(outputs, input_size_q, use_bias=True) # [batch_size, sequence_len_q, d_model].
        return outputs

def masking(inputs):
    # d_q = d_k
    mask = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1))) # [batch_size, sequence_len_k]
    mask = tf.tile(mask, [1, 1]) # [batch_size, sequence_len_k]
    mask = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(inputs)[1], 1]) # [batch_size, sequence_len_k, sequence_len_k]  
    return mask 

def mask_seq(inputs, dim):
    '''
    input: input sequences
    output: masks
    '''
    mask = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1))) # [batch_size, sequence_len_k]
    mask = tf.tile(tf.expand_dims(mask, 1), [1, dim, 1])
    mask = tf.transpose(mask, [0, 2, 1])
    return mask 

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    if inputs.get_shape().ndims == 2:
        shape = tf.shape(tf.reduce_mean(inputs, 0))
        epsilon = 1e-3
        scale = tf.Variable(tf.ones(shape))
        beta = tf.Variable(tf.zeros(shape))
        pop_mean = tf.Variable(tf.zeros(shape), trainable=False)
        pop_var = tf.Variable(tf.ones(shape), trainable=False)
        
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)
    else:
        raise NotImplementedError

def label_smoothing(inputs):
  K = inputs.get_shape().as_list()[-1]
  label = ((1-0.1) * inputs) + (0.1 / K)
  return label

def layer_norm(x, keep_proba, is_training=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('norm', reuse=reuse):
        if is_training:
            x = tf.nn.dropout(x, keep_proba)
        # Normalize
        x = tf.contrib.layers.layer_norm(x)
    return x

def evaluation(groundtruth, prediction):

    ave_uar = recall_score(groundtruth, prediction, average='macro')
    ave_acc = accuracy_score(groundtruth, prediction)
    conf = confusion_matrix(groundtruth, prediction)
    print('Ave test acc: {:.3f}, Ave test uar: {:.3f}'.format(ave_acc, ave_uar))   
    print(conf)
    
    return ave_uar, ave_acc
