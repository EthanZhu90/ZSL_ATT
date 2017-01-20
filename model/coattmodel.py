import tensorflow as tf
import math
#import sys
#sys.path.append("/home/ethan/research/MCB_VQA/model/")
#from vqamodel import *


class coattention_model():
    def __init__(self, batch_size, feature_dim, embed_dim, n_lstm_steps, word_num, phrs_size_all, phrs_size_stepByCls):

        self.margin = 100. ## for max margin loss
        self.hidden_dim_v = 256
        self.parts_Num = 7
        self.feature_dim = 512
        self.phrs_Num = 100
        self.batch_size = batch_size
        self.batch_phrs_size = self.batch_size * self.phrs_Num
        self.hidden_dim_u =512

        self.embed_dim = embed_dim
        self.n_lstm_steps = n_lstm_steps
        self.lstm_hidden_dim = 512
        self.lstm_feat_dim = self.lstm_hidden_dim * 2
        #self.phrase_size = phrs_size_all  # # of phrases for all classes
        self.phrs_size_stepByCls = phrs_size_stepByCls
        self.word_num = word_num  # # of word in dictionary 1156
        self.class_num = 150
        self.l2_weight = 0.01

        # self.proj_dim
        # Word Embedding E (K*m)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform(
                [self.word_num, self.embed_dim], -1.0, 1.0), name='Wemb')
        self.init_hidden_W = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim], name='init_hidden_W')
        self.init_hidden_b = self.init_bias([self.lstm_hidden_dim],
                                            name='init_hidden_bias')
        self.init_memory_W = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim], name='init_memory_W')
        self.init_memory_b = self.init_bias([self.lstm_hidden_dim],
                                            name='init_memory_bias')

        self.lstm_W1 = self.init_weight(
            [self.embed_dim, self.lstm_hidden_dim * 4], name='lstm_W1')
        self.lstm_W2 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_W2')
        self.lstm_U1 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_U1')
        self.lstm_U2 = self.init_weight(
            [self.lstm_hidden_dim, self.lstm_hidden_dim * 4], name='lstm_U2')
        self.lstm_b1 = self.init_bias([self.lstm_hidden_dim * 4],
                                      name='lstm_bias_1')
        self.lstm_b2 = self.init_bias([self.lstm_hidden_dim * 4],
                                      name='lstm_bias_2')

        self.w_v_1 = self.init_weight(
            [self.feature_dim, self.hidden_dim_v], name='w_v_1')
        self.W_vm_1 = self.init_weight(
            [self.feature_dim, self.hidden_dim_v], name='W_vm_1')
        self.W_vh_1 = self.init_weight(
            [self.hidden_dim_v, 1], name='W_vh_1')

        self.w_u_1 = self.init_weight(
            [self.lstm_feat_dim, self.hidden_dim_u], name='w_u_1')
        self.w_um_1 = self.init_weight(
            [self.lstm_feat_dim, self.hidden_dim_u], name='w_um_1')
        self.W_uh_1 = self.init_weight(
            [self.hidden_dim_u, 1], name='W_uh_1')
        self.P_1 = self.init_weight(
            [self.feature_dim, self.lstm_feat_dim], name='P_1')

    def init_weight(self, shape, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal(shape,
                stddev=stddev/math.sqrt(float(shape[0]))), name=name)

    def init_bias(self, shape, name=None):
        return tf.Variable(tf.zeros(shape), name=name)

    def get_initial_lstm(self):
        return tf.zeros([self.batch_phrs_size, self.lstm_hidden_dim]), \
                tf.zeros([self.batch_phrs_size, self.lstm_hidden_dim])

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
                    word_embed = tf.zeros([self.batch_phrs_size, self.embed_dim])
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        word_embed = \
                            tf.nn.embedding_lookup(self.Wemb, phrase[:, idx - 1])
                h1, c1, h2, c2 = self.forward_prop(h1, c1, h2, c2, word_embed, reuse=idx)
        return tf.concat(1, [h1, h2])


    def get_visual_att(self, image_feat):
        ## visual attention
        m_v_0 = tf.reduce_mean(image_feat, axis=1)
        image_feat_2d = tf.reshape(image_feat, [-1, self.feature_dim]) ## 2st dim = self.feature_dim
        WV = tf.reshape(tf.tanh(tf.matmul( image_feat_2d, self.w_v_1)),
                        [self.batch_size, self.parts_Num, self.hidden_dim_v])
        WM = tf.tanh(tf.matmul(m_v_0, self.W_vm_1))  ## batch_size, hidden_dim_v

        h_v_1 = tf.multiply(WV,
                            tf.tile(tf.expand_dims(WM, 1), [1, self.parts_Num, 1]))  # element-wise mul
        MH = tf.reshape(tf.matmul(tf.reshape(h_v_1, [-1, self.hidden_dim_v]), self.W_vh_1),
                        [self.batch_size, self.parts_Num, 1])

        alpha_v = tf.nn.softmax(MH, dim=1)
        weighted_V = tf.tanh( tf.matmul( tf.reshape(tf.batch_matmul(tf.transpose(image_feat, [0, 2, 1]), alpha_v),
                                                    [self.batch_size, self.feature_dim]),
                                         self.P_1 ))

        return weighted_V
    def get_text_att(self, phrase_feat):

        ## first get embedding
        phrase_embed_2d = self.phrase_embed(tf.reshape(phrase_feat, [-1, self.n_lstm_steps]))
        phrase_embed = tf.reshape(phrase_embed_2d, [self.batch_size, self.phrs_Num, -1])

        m_u_0 = tf.reduce_mean(phrase_embed, axis=1)
        WU = tf.reshape(tf.tanh(tf.matmul(phrase_embed_2d, self.w_u_1)),
                        [self.batch_size, self.phrs_Num, self.hidden_dim_u])
        WM = tf.tanh(tf.matmul(m_u_0, self.w_um_1))  ## batch_size, hidden_dim_v

        h_u_1 = tf.multiply(WU,
                            tf.tile(tf.expand_dims(WM, 1), [1, self.phrs_Num, 1]))  # element-wise mul
        MU = tf.reshape(tf.matmul(tf.reshape(h_u_1, [-1, self.hidden_dim_u]), self.W_uh_1),
                        [self.batch_size, self.phrs_Num, 1])
        alpha_u = tf.nn.softmax(MU, dim=1)

        weighted_U = tf.tanh(tf.reshape(tf.batch_matmul(tf.transpose(phrase_embed, [0, 2, 1]), alpha_u),
                                        [self.batch_size, self.lstm_feat_dim]))
        return weighted_U
    def get_sim(self, weighted_V, weighted_U):

        return tf.reshape( tf.batch_matmul(tf.expand_dims(weighted_V, dim=1),
                                           tf.expand_dims(weighted_U, dim=2)),
                           [self.batch_size, 1])

    def model(self):
        image_feat     = tf.placeholder("float32", [self.batch_size, self.parts_Num, self.feature_dim])
        image_feat_neg = tf.placeholder("float32", [self.batch_size, self.parts_Num, self.feature_dim])
        phrase_feat     = tf.placeholder("int32",  [self.batch_size, self.phrs_Num,  self.n_lstm_steps])
        phrase_feat_neg = tf.placeholder("int32",  [self.batch_size, self.phrs_Num,  self.n_lstm_steps])

        weighted_V     = self.get_visual_att(image_feat)
        weighted_V_neg = self.get_visual_att(image_feat_neg)

        weighted_U     = self.get_text_att(phrase_feat)
        weighted_U_neg = self.get_text_att(phrase_feat_neg)

        sim_VU  = self.get_sim(weighted_V, weighted_U)
        sim_V_U = self.get_sim(weighted_V_neg, weighted_U)
        sim_VU_ = self.get_sim(weighted_V, weighted_U_neg)
        vars = tf.trainable_variables()

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * self.l2_weight

        loss = tf.add( tf.reduce_mean(
                            tf.maximum(0., self.margin - sim_VU + sim_V_U) + \
                            tf.maximum(0., self.margin - sim_VU + sim_VU_)),
                       lossL2)

        acc_tmp = tf.logical_and( tf.greater(sim_VU, sim_V_U), tf.greater(sim_VU, sim_VU_) )
        acc = tf.reduce_mean( tf.to_float(acc_tmp) )

        return image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, weighted_V, weighted_U, loss, acc


    def trainer(self):
        image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, _, _, loss, acc = self.model()

        return image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, loss, acc

    def solver(self):
        image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, weighted_V, weighted_U, _, _ = self.model()

        return image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, weighted_V, weighted_U

if __name__ == '__main__':
    model = attention_model(128, [2048], 16000, 20000, 300, 3000, 50)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()