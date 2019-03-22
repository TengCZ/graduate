# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n = 2):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    inx = np.argsort(p)[-top_n:]
    value = np.sort(p)[-top_n:]
    res = []
    for i in range(top_n):
        res.append((inx[i],value[i]))
    res.sort(key=lambda x: x[1], reverse=True)
    return res

def attention(inputs, attention_size=50):

    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        att_inputs = tf.concat(2, inputs)
    else:
        att_inputs = inputs
    sequence_length = att_inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = att_inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(att_inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(att_inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    return output




class CharRNN:
    def __init__(self, num_classes, num_seqs=128, num_steps=50,
                 lstm_size=128, num_layers=1, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, embedding_size=256):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
        ####print log
        print('num_class :',self.num_classes)
        print('num_seqs :',self.num_seqs)
        print('num_steps :',self.num_steps)
        print('lstm_size :',self.lstm_size)
        print('num_layers :',self.num_layers)
        print('learning_rating :',self.learning_rate)
        print('grad_clip :',self.grad_clip)
        print('train_keep_prob :',self.train_keep_prob)
        print('embeding_size :',self.embedding_size)
        print('sample :',sampling)

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            with tf.device("/cpu:0"):
                  embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                  self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            # lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            lstm = tf.contrib.rnn.GRUCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            self.attention_outputs = attention(self.lstm_outputs)

            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])
            # x = tf.matmul(x,self.attention_outputs)


            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes) )
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())

            correct_prediction = tf.equal(tf.argmax(self.proba_prediction, 1), tf.argmax(y_reshaped, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, save_path):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())

            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                accuracy,batch_loss, new_state, _ = sess.run([self.accuracy,self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)
                end = time.time()

                print('batch_loss: {:.4f}... '.format(batch_loss),
                          'batch_accuracy:{:.4f}...'.format(accuracy),
                          'sec/batch:{:.4f} '.format((end - start)))

            save_path = os.path.join(save_path, 'model')
            self.saver.save(sess, save_path)
            print('save to the',save_path)

    def test(self, batch_generator):
        sess = self.session
        new_state = sess.run(self.initial_state)
        for x,y in batch_generator:
            test_accuracy_list = []
            test_loss_list = []
            feed = {self.inputs: x,
                    self.targets: y,
                    self.keep_prob: 1,
                    self.initial_state: new_state}
            accuracy, batch_loss, new_state, _ = sess.run([self.accuracy, self.loss,
                                                           self.final_state,
                                                           self.optimizer],
                                                          feed_dict=feed)
            print('batch_loss: {:.4f}... '.format(batch_loss),
                  'batch_accuracy:{:.4f}...'.format(accuracy))
            test_accuracy_list.append(accuracy)
            test_loss_list.append(batch_loss)
        return np.mean(test_accuracy_list),np.mean(test_loss_list)


    def sample(self,prime, vocab_size):
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
        tokens = pick_top_n(preds, vocab_size)
        return tokens

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
