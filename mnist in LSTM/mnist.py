# coding=utf-8
import os
os.chdir("../")
import tensorflow as tf
import model
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def train(model, sess, training_iters, display_step):
    train_writer = tf.summary.FileWriter('./mnist in LSTM/train', sess.graph)
    sess.run(init)
    step = 0
    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(model.batch_size)
        batch_x = batch_x.reshape((model.batch_size, model.steps, model.inputs))
        sess.run(model.optimizer, feed_dict={model.x: batch_x, model.y: batch_y, model.keep_prob: 0.5})
        if step % display_step == 0:
            summary, acc, loss = sess.run([model.merged, model.accuracy, model.cross_entropy],
                                          feed_dict={model.x: batch_x,
                                                     model.y: batch_y,
                                                     model.keep_prob: 1.0})
            train_writer.add_summary(summary, step)
            print "step %d, training accuracy %g, cross entropy %g" % (step, acc, loss)
            test(model, sess)
        step += 1
    print("Optimization Finished!")


def test(model, sess):
    test_data = mnist.test.images[:model.batch_size]
    test_label = mnist.test.labels[:model.batch_size]
    test_data = test_data.reshape((model.batch_size, model.steps, model.inputs))
    acc = sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label, model.keep_prob: 1.0})
    print "test accracy %g" % acc


def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./mnist in LSTM/model/model.ckpt")
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    training_iters = 1000
    display_step = 10
    my_network = model.LSTM_layer(name="mnist")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        train(my_network, sess, training_iters, display_step)
        test(my_network, sess)
        save(sess)