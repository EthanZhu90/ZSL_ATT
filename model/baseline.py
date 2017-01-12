import tensorflow as tf
import numpy as np
import math


class baseline_model():
    def __init__(self, batch_size, feature_dim, embed_dim, n_lstm_steps, word_num, phrs_size_all, phrs_size_stepByCls):
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.n_lstm_steps = n_lstm_steps
        self.lstm_hidden_dim = 1024
        self.phrase_size = phrs_size_all # # of phrases for all classes
        self.phrs_size_stepByCls = phrs_size_stepByCls
        self.word_num = word_num # # of word in dictionary 1156
        self.class_num = 150
        self.feats_dim = 3584

        #self.proj_dim
        # Word Embedding E (K*m)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform(
                [self.word_num, self.embed_dim], -1.0, 1.0), name='Wemb')
        self.init_hidden_W = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim], name='init_hidden_W')
        self.init_hidden_b = self.init_bias([self.lstm_hidden_dim],
                                            name='init_hidden_b')
        self.init_memory_W = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim], name='init_memory_W')
        self.init_memory_b = self.init_bias([self.lstm_hidden_dim],
                                            name='init_memory_b')


        self.lstm_W1 = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim * 4], name='lstm_W1')
        self.lstm_W2 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_W2')
        self.lstm_U1 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_U1')
        self.lstm_U2 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_U2')
        self.lstm_b1 = self.init_bias([self.lstm_hidden_dim * 4],
                                      name='lstm_b1')
        self.lstm_b2 = self.init_bias([self.lstm_hidden_dim * 4],
                                      name='lstm_b2')
        #self.fc_W = self.init_weight([self.proj_dim, self.ans_candi_num],
        #                                     name='fc_W')
        #self.fc_b = self.init_bias([self.ans_candi_num], name='fc_b')

        self.Wxt = self.init_weight([self.feats_dim, 1024],
                                            name='Wxt')



    def init_weight(self, shape, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal(shape,
                stddev=stddev/math.sqrt(float(shape[0]))), name=name)

    def init_bias(self, shape, name=None):
        return tf.Variable(tf.zeros(shape), name=name)

    def get_initial_lstm(self):
        return tf.zeros([self.phrase_size, self.lstm_hidden_dim]), \
                tf.zeros([self.phrase_size, self.lstm_hidden_dim])

    def forward_prop(self, h1, c1, h2, c2, word_emb, is_train=True, reuse=False):
        lstm_preactive1 = tf.matmul(h1, self.lstm_U1) + \
                            tf.matmul(word_emb, self.lstm_W1) + self.lstm_b1
        i1, f1, o1, g1 = tf.split(1, 4, lstm_preactive1)
        i1 = tf.nn.sigmoid(i1)
        f1 = tf.nn.sigmoid(f1)
        o1 = tf.nn.sigmoid(o1)
        g1 = tf.nn.tanh(g1)
        c1 = f1*c1 + i1*g1
        h1 = o1*tf.nn.tanh(c1)

        lstm_preactive2 = tf.matmul(h2, self.lstm_U2) + \
                            tf.matmul(h1, self.lstm_W2) + self.lstm_b2
        i2, f2, o2, g2 = tf.split(1, 4, lstm_preactive2)
        i2 = tf.nn.sigmoid(i2)
        f2 = tf.nn.sigmoid(f2)
        o2 = tf.nn.sigmoid(o2)
        g2 = tf.nn.tanh(g2)
        c2 = f2*c2 + i2*g2
        h2 = o2*tf.nn.tanh(c2)

        return h1, c1, h2, c2

    def phrase_embed(self, phrase):
        h1, c1 = self.get_initial_lstm()
        h2, c2 = self.get_initial_lstm()
        with tf.variable_scope('lstm'):
            for idx in range(self.n_lstm_steps):
                if idx == 0:
                    word_embed = tf.zeros([self.phrase_size, self.embed_dim])
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        word_embed = \
                            tf.nn.embedding_lookup(self.Wemb, phrase[:, idx - 1])
                h1, c1, h2, c2 = self.forward_prop(h1, c1, h2, c2, word_embed, reuse=idx)
        return tf.concat(1, [h1, h2])

    def get_loss(self, pred_score, groundtruth):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_score, groundtruth, name=None)
        return tf.reduce_sum(cross_entropy) / self.batch_size

    '''
    def get_loss(self, logit, answer):
        label = tf.expand_dims(answer, 1)
        index = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        concated = tf.concat(1, [index, label])
        onehot_label = tf.sparse_to_dense(concated,
                tf.pack([self.batch_size, self.ans_candi_num]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit, onehot_label)
        return tf.reduce_sum(cross_entropy)/self.batch_size
    '''

    def model(self):
        image_feat = tf.placeholder("float32", [self.batch_size, self.feature_dim])
        phrase_feat = tf.placeholder("int32", [self.phrase_size, self.n_lstm_steps])
        labels = tf.placeholder("int32", [self.batch_size])


        #feat = tf.concat(1, [image_feat, phrase_feat])
        #normalized_feat = tf.nn.l2_normalize(feat, 0)  ## do I need to add this?

        phrase_embed = self.phrase_embed(phrase_feat)

        cls_weight = tf.matmul(self.Wxt, tf.transpose(phrase_embed))
        score_mat = tf.matmul(image_feat, cls_weight)

        #score_mat
        #start_end = zip(self.phrs_size_stepByCls[0:-2], self.phrs_size_stepByCls[1:-1])
        #score = np.array(map(lambda x, numpy.sum(score_mat[:, x[0]:x[1]], axis=0), start_end))
        pred_score = np.empty((self.batch_size,0), dtype= 'float32')
        for start, end in zip(self.phrs_size_stepByCls[0:-1], self.phrs_size_stepByCls[1:]):
            if(start == 0):
                # get average over all phrases per class
                pred_score = tf.div(tf.reduce_sum(score_mat[:, start:end], axis = 1, keep_dims=True), (end-start))
            else:
                tmp_score = tf.div(tf.reduce_sum(score_mat[:, start:end], axis=1, keep_dims=True), (end - start))
                pred_score = tf.concat(1, [pred_score, tmp_score])

        return image_feat, phrase_feat, labels,  pred_score



    def trainer(self):
        image_feat, phrase_feat, labels, pred_score = self.model()
        loss = self.get_loss(pred_score, labels)
        max_prob_words = tf.argmax(pred_score, 1)
        max_prob_words = tf.cast(max_prob_words, tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(max_prob_words, labels), tf.float32))

        return image_feat, phrase_feat, labels, pred_score, loss, acc

    def solver(self):
        image_feat, phrase_feat, labels, pred_score = self.model()
        max_prob_words = tf.argmax(pred_score, 1)

        return image_feat, phrase_feat, labels, max_prob_words

if __name__ == '__main__':
    model = baseline_model(21, 512*7, 200, 4)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()