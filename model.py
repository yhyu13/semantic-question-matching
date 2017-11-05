import tensorflow as tf
from tqdm import tqdm
import os
from data_utils import Data_iterator

max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()


class SiameseNet(object):
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings

    def bilstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, sequence_length=seq_len,
                                                                    dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output

    def activation(self, x):
        assert self.config.fd_activation in ["sigmoid", "relu", "tanh"]
        if self.config.fd_activation == "sigmoid":
            return tf.nn.sigmoid(x)
        elif self.config.fd_activation == "relu":
            return tf.nn.relu(x)
        elif self.config.fd_activation == "tanh":
            return tf.nn.tanh(x)

    def build(self):
        ### Placeholders
        self.q1 = tf.placeholder(tf.int64, shape=[None, None], name="question1")
        self.l1 = tf.placeholder(tf.int64, shape=[None], name="len1")

        self.q2 = tf.placeholder(tf.int64, shape=[None, None], name="question2")
        self.l2 = tf.placeholder(tf.int64, shape=[None], name="len2")

        self.y = tf.placeholder(tf.int64, shape=[None], name="is_duplicate")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        ### Embedding layer
        with tf.variable_scope("word_embeddings") as scope:
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            we1 = tf.nn.embedding_lookup(_word_embeddings, self.q1, name="q1_embedded")
            we2 = tf.nn.embedding_lookup(_word_embeddings, self.q2, name="q2_embedded")

            we1 = tf.nn.dropout(we1, self.dropout)
            we2 = tf.nn.dropout(we2, self.dropout)

        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1 = self.bilstm(we1, self.l1)
            scope.reuse_variables()
            lstm2 = self.bilstm(we2, self.l2)

        ### Max pooling
        lstm1_pool = max_pool(lstm1)
        lstm2_pool = max_pool(lstm2)

        ### Features
        flat1 = tf.contrib.layers.flatten(lstm1_pool)
        flat2 = tf.contrib.layers.flatten(lstm2_pool)
        mult = tf.multiply(flat1, flat2)
        diff = tf.abs(tf.subtract(flat1, flat2))
        concat = tf.concat([flat1, flat2, mult, diff], axis=-1)

        ### FC layers
        intermediary_size = 2 + (8 * self.config.hidden_size - 2) // 2

        with tf.variable_scope("fc1") as scope:
            W1 = tf.Variable(tf.random_normal([8 * self.config.hidden_size, intermediary_size]), name="w_fc")
            b1 = tf.Variable(tf.zeros([intermediary_size]), name="b_fc")

            preact1 = tf.matmul(concat, W1) + b1
            fc1 = self.activation(preact1)

        with tf.variable_scope("fc2") as scope:
            W2 = tf.Variable(tf.random_normal([intermediary_size, 2]), name="w_fc")
            b2 = tf.Variable(tf.zeros([2]), name="b_fc")

            preact2 = tf.matmul(fc1, W2) + b2
            self.fc2 = preact2

        ### Loss
        self.cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc2))

        ### Optimizer
        with tf.variable_scope("train_step") as scope:
            if self.config.lr_method == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.lr_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.lr)

            self.train_step = optimizer.minimize(self.cross_entropy)

        ### Evaluation
        correct_prediction = tf.equal(tf.argmax(self.fc2, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ### Init
        self.init = tf.global_variables_initializer()

        ### Summaries
        with tf.variable_scope("summaries") as scope:

            # train
            tf.summary.scalar('cross_entropy', self.cross_entropy)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()

            # test
            self.acc_value = tf.placeholder_with_default(tf.constant(0.0), shape=())
            self.ce_value = tf.placeholder_with_default(tf.constant(0.0), shape=())
            acc_summary = tf.summary.scalar('accuracy', self.acc_value)
            ce_summary = tf.summary.scalar('cross_entropy', self.ce_value)
            self.merged_eval = tf.summary.merge([acc_summary, ce_summary])

        self.saver = tf.train.Saver()

    def run_epoch(self, sess, train, dev, test, epoch):
        iterator = Data_iterator(train, self.config.batch_size)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size

        for i in tqdm(range(nbatches)):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.q2: q2, self.l1: l1, self.l2: l2, self.y: y,
                  self.dropout: self.config.dropout,
                  self.lr: self.config.lr}
            _, summary, acc = sess.run([self.train_step, self.merged, self.accuracy], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.train_writer.add_summary(summary, epoch * nbatches + i)
                # except tf.errors.OutOfRangeError:
                #     break

        summary, dev_acc = self.run_evaluate(sess, dev)
        self.dev_writer.add_summary(summary, epoch * nbatches + i)
        print("dev acc {:04.2f}".format(100 * dev_acc))

        summary, test_acc = self.run_evaluate(sess, test)
        self.test_writer.add_summary(summary, epoch * nbatches + i)
        print("test acc {:04.2f}".format(100 * test_acc))

        return dev_acc

    def run_evaluate(self, sess, data):
        iterator = Data_iterator(data, self.config.batch_size)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size

        accuracy, cross = 0, 0
        for i in range(nbatches):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.q2: q2, self.l1: l1, self.l2: l2, self.y: y,
                  self.dropout: 0,
                  self.lr: 0}
            acc, ce = sess.run([self.accuracy, self.cross_entropy], feed_dict=fd)

            accuracy += acc * len(q1)
            cross += ce * len(q1)

        accuracy /= iterator.max
        cross /= iterator.max

        summary = sess.run(self.merged_eval, feed_dict={self.acc_value: accuracy, self.ce_value: cross})

        return summary, accuracy

    def train(self, train_data, dev_data, test_data):

        best_acc = 0
        nepoch_no_improv = 0

        with tf.Session() as sess:
            self.train_writer = tf.summary.FileWriter(self.config.log_path + "train", sess.graph)
            self.dev_writer = tf.summary.FileWriter(self.config.log_path + "dev", sess.graph)
            self.test_writer = tf.summary.FileWriter(self.config.log_path + "test", sess.graph)

            sess.run(self.init)

            print("Training in {}".format(self.config.conf_dir))
            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{} :".format(epoch + 1, self.config.n_epochs))
                dev_acc = self.run_epoch(sess, train_data, dev_data, test_data, epoch)

                self.config.lr *= self.config.lr_decay

                if dev_acc >= best_acc:
                    nepoch_no_improv = 0
                    if not os.path.exists(self.config.model_path):
                        os.makedirs(self.config.model_path)
                    self.saver.save(sess, self.config.model_path)
                    best_acc = dev_acc
                    print("New best score on dev !")

                else:
                    nepoch_no_improv += 1
                    if nepoch_no_improv >= self.config.nepochs_no_improv:
                        print("Early stopping after {} epochs without improvements".format(nepoch_no_improv))
                        break
