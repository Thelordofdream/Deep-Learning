import tensorflow as tf
from tensorflow.contrib import rnn


class Bd_LSTM_layer(object):
    def __init__(self, name="X", steps=28, inputs=28, hidden=28, batch_size=100, classes=10, learning_rate=0.001):
        self.name = name

        self.steps = steps
        self.inputs = inputs
        self.hidden = hidden
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate

        self.output = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.create_nerual_network()

    def create_nerual_network(self):
        self.x = tf.placeholder("float", [None, self.steps, self.inputs], name="x")
        with tf.variable_scope("input_layer"):
            input = self.shape_tranform()

        with tf.variable_scope("Bd_LSTM_layer"):
            _seq_len = tf.fill([self.batch_size], tf.constant(self.steps, dtype=tf.float32))
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

        with tf.variable_scope("dense_layer"):
            outputs = tf.transpose(outputs, [1, 0, 2])
            time_seq = tf.reshape(outputs, [-1, self.steps * 2 * self.inputs])
            hidden1_w = tf.Variable(tf.random_normal([self.steps * 2 * self.inputs, self.hidden]), name='h1_w')
            hidden1_b = tf.Variable(tf.random_normal([self.hidden]), name='h1_b'),
            h1 = tf.matmul(time_seq, hidden1_w) + hidden1_b

        with tf.variable_scope("dropout"):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            h_drop = tf.nn.dropout(h1, self.keep_prob)

        with tf.variable_scope("readout_layer"):
            hidden2_w = tf.Variable(tf.random_normal([self.hidden, self.classes]), name='h2_w')
            hidden2_b = tf.Variable(tf.random_normal([self.classes]), name='h2_b')
            self.output = tf.matmul(h_drop, hidden2_w) + hidden2_b

        self.y = tf.placeholder("float", [None, self.classes], name="y")
        with tf.variable_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
            correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def shape_tranform(self):
        X = tf.transpose(self.x, [1, 0, 2])
        X = tf.reshape(X, [-1, self.inputs])
        X = tf.split(X, self.steps, 0)
        return X