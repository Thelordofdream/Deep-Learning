import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class Attensive_Reader(object):
    def __init__(self, name="X", steps=28, inputs=28, hidden_q=28, batch_size=200, classes=10, learning_rate=0.001):
        self.name = name

        self.steps = steps
        self.inputs = inputs
        self.hidden_q = hidden_q
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate

        self.output = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.create_nerual_network()

    def create_nerual_network(self):
        self.q = tf.placeholder("float", [None, self.steps, self.inputs], name="q")
        with tf.variable_scope("input_layer_q"):
            input_q = self.shape_tranform(self.q, self.steps)

        with tf.variable_scope("Q_LSTM_layer"):
            outputs_q, output1_q, output2_q = self.create_LSTM_layer(input_q, seq_len=self.steps)

        with tf.variable_scope("hidden_layer_q"):
            outputs_q = tf.reshape(outputs_q, [-1, 2 * self.inputs])  # (n_steps*batch_size, n_input)
            hidden1_q_w = tf.Variable(tf.random_normal([2 * self.inputs, 2 * self.hidden_q]), name='hq1_w')
            hidden1_q_b = tf.Variable(tf.random_normal([2 * self.hidden_q]), name='hq1_b'),
            hq_1 = tf.matmul(outputs_q, hidden1_q_w) + hidden1_q_b
            hq_1 = tf.split(hq_1, self.steps, 0)

        with tf.variable_scope("dropout_q"):
            self.keep_prob_q = tf.placeholder(tf.float32, name="keep_prob_q")
            hq1_drop = tf.nn.dropout(hq_1, self.keep_prob_q)

        self.a = tf.placeholder("float", [None, self.steps, self.inputs], name="a")
        with tf.variable_scope("input_layer_a"):
            input_a = self.shape_tranform(self.a, self.steps)

        with tf.variable_scope("A_LSTM_layer"):
            outputs_a, output1_a, output2_a = self.create_LSTM_layer(input_a, seq_len=self.steps)

        with tf.variable_scope("hidden_layer_a_fw"):
            hidden1_a_w = tf.Variable(tf.random_normal([self.inputs, self.hidden_q]), name='ha1_w')
            hidden1_a_b = tf.Variable(tf.random_normal([self.hidden_q]), name='ha1_b')
            output1_a = tf.reshape(output1_a, [-1, self.inputs])
            ha1_fw = tf.matmul(output1_a, hidden1_a_w) + hidden1_a_b
            ha1_fw = tf.split(ha1_fw, 2, 0)

        with tf.variable_scope("hidden_layer_a_bw"):
            output2_a = tf.reshape(output2_a, [-1, self.inputs])
            ha1_bw = tf.matmul(output2_a, hidden1_a_w) + hidden1_a_b
            ha1_bw = tf.split(ha1_bw, 2, 0)

        ha_1 = tf.concat([ha1_fw[0], ha1_bw[0]], 1, name="concat")

        with tf.variable_scope("dropout_a"):
            self.keep_prob_a = tf.placeholder(tf.float32, name="keep_prob_a")
            ha1_drop = tf.nn.dropout(ha_1, self.keep_prob_a)

        with tf.variable_scope("attention_layer"):
            Wum = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wum')
            mu = tf.matmul(ha_1, Wum)
            m = []
            Wym = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wym')
            for i in range(self.steps):
                m.append(tf.nn.tanh(tf.matmul(hq1_drop[i], Wym) + mu))
            m = tf.reshape(m, [-1, 2 * self.hidden_q])
            Wms = tf.Variable(tf.random_normal([2 * self.hidden_q, 1]), name='Wms')
            self.s = tf.placeholder("float", [None, self.steps], name="s")
            s0 = tf.matmul(m, Wms)
            s0 = tf.split(s0, self.steps, 0)
            hq1_drop = tf.transpose(hq1_drop, [1, 0, 2])
            s0 = tf.reshape(s0, [self.batch_size, 28])
            s0 = tf.nn.softmax(s0)
            self.s = tf.reshape(s0, [self.batch_size, 28, 1])
            r = []
            for i in range(self.batch_size):
                r.append(tf.transpose(tf.matmul(tf.transpose(hq1_drop[i]), self.s[i])))
            r = tf.reshape(r, [-1, 2 * self.hidden_q])

        with tf.variable_scope("keyword_layer"):
            Wug = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wug')
            Wrg = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wrg')
            g = tf.nn.tanh(tf.matmul(ha1_drop, Wug) + tf.matmul(r, Wrg))
            # self.output = tf.matmul(ha1_drop, Wug) + tf.matmul(r, Wrg)

        with tf.variable_scope("read_out"):
            Wg = tf.Variable(tf.random_normal([2 * self.hidden_q, self.classes]), name='ha1_w')
            bg = tf.Variable(tf.random_normal([self.classes]), name='ha1_b')
            self.output= tf.matmul(g, Wg) + bg

        self.y = tf.placeholder("float", [None, self.classes], name="y")
        with tf.variable_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
            correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def shape_tranform(self, x, steps):
        X = tf.transpose(x, [1, 0, 2])
        X = tf.reshape(X, [-1, self.inputs])
        X = tf.split(X, steps, 0)
        return X

    def create_LSTM_layer(self, input, seq_len):
        _seq_len = tf.fill([self.batch_size], tf.constant(seq_len, dtype=tf.float32))
        with tf.variable_scope("Forward_LSTM"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.inputs, forget_bias=0.1, state_is_tuple=True)
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        with tf.variable_scope("Backward_LSTM"):
            lstm_bw_cell = rnn.BasicLSTMCell(self.inputs, forget_bias=0., state_is_tuple=True)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        outputs, output1, output2 = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                                                 initial_state_fw=lstm_fw_cell.zero_state(
                                                                     self.batch_size, tf.float32),
                                                                 initial_state_bw=lstm_bw_cell.zero_state(
                                                                     self.batch_size, tf.float32),
                                                                 sequence_length=_seq_len)
        return outputs, output1, output2
