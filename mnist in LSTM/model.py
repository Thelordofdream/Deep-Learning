import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM_layer(object):
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

        with tf.variable_scope("LSTM_layer"):
            lstm_cell = rnn.BasicLSTMCell(self.inputs, forget_bias=0., state_is_tuple=True)
            lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0,
                                              seed=None)
            outputs, states = rnn.static_rnn(lstm_cell, input, initial_state=lstm_cell.zero_state(self.batch_size, tf.float32))

        with tf.variable_scope("dense_layer"):
            hidden1_w = tf.Variable(tf.random_normal([self.hidden, self.classes]), name='h1_w')
            hidden1_b = tf.Variable(tf.random_normal([self.classes]), name='h1_b'),
            h1 = tf.matmul(outputs[-1], hidden1_w) + hidden1_b

        with tf.variable_scope("dropout"):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.output = tf.nn.dropout(h1, self.keep_prob)

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