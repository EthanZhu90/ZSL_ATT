import tensorflow as tf
import math
#import sys
#sys.path.append("/home/ethan/research/MCB_VQA/model/")
#from vqamodel import *


class coattention_model2():
    def __init__(self, batch_size, feature_dim, embed_dim, n_lstm_steps, word_num, phrs_size_all, phrs_size_stepByCls, class_num):

        self.margin = 100. ## for max margin loss
        self.hidden_dim_v = 256
        self.parts_Num = 7
        self.feature_dim = 512
        self.phrs_Num = 100
        self.batch_size = batch_size

        self.hidden_dim_u =512
        self.batch_cls_Num = 0 ## dummy num
        self.batch_phrs_size = 0 #self.batch_cls_Num * self.phrs_Num

        self.embed_dim = embed_dim
        self.n_lstm_steps = n_lstm_steps
        self.lstm_hidden_dim = 512
        self.lstm_feat_dim = self.lstm_hidden_dim * 2
        #self.phrase_size = phrs_size_all  # # of phrases for all classes
        self.phrs_size_stepByCls = phrs_size_stepByCls
        self.word_num = word_num  # # of word in dictionary 1156
        self.class_num = class_num
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
            [self.lstm_feat_dim, self.hidden_dim_v], name='W_vm_1')
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
        self.P_0 = self.init_weight(
            [self.feature_dim, self.lstm_feat_dim], name='P_0')

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


    def get_visual_att(self, image_feat, m_v_0):

        ## visual attention
        image_feat_2d = tf.reshape(image_feat, [-1, self.feature_dim]) ## 2st dim = self.feature_dim
        WV = tf.reshape(tf.tanh(tf.matmul( image_feat_2d, self.w_v_1)),
                        [self.batch_size, self.parts_Num, self.hidden_dim_v])

        WM = tf.tanh( tf.reshape( tf.matmul( tf.reshape(m_v_0, [-1, self.lstm_feat_dim]),
                                              self.W_vm_1),
                                  [self.batch_size, self.class_num, self.hidden_dim_v]) )  ## batchSize * ClsNum , hidden_dim_v


        h_v_1 = tf.multiply(tf.tile(tf.expand_dims(WV, 1), [1, self.class_num, 1, 1]),
                            tf.tile(tf.expand_dims(WM, 2), [1, 1, self.parts_Num, 1]) )   # element-wise mul
        MH = tf.reshape(tf.matmul(tf.reshape(h_v_1, [-1, self.hidden_dim_v]), self.W_vh_1),
                        [self.batch_size, self.class_num, self.parts_Num, 1])

        alpha_v = tf.nn.softmax(MH, dim=2)

        weighted_V = tf.batch_matmul(tf.transpose(tf.tile(tf.expand_dims(image_feat, 1), [1, self.class_num, 1, 1]), [0, 1, 3, 2]),
                                    alpha_v)

        weighted_V_final = tf.tanh( tf.reshape( tf.matmul( tf.reshape( weighted_V, [-1, self.feature_dim]),
                                                  self.P_1 ),
                                       [self.batch_size, self.class_num, self.lstm_feat_dim]) )

        return weighted_V_final

    def get_text_att(self, lstm_feat_2d, lstm_feat, m_u_0):

        ## self.batch_cls_Num is 32 in att1 or 150 in att2

        WU = tf.reshape(tf.tanh(tf.matmul(lstm_feat_2d, self.w_u_1)),
                        [self.batch_cls_Num, self.phrs_Num, self.hidden_dim_u])

        WM = tf.tanh( tf.reshape( tf.matmul(tf.reshape(m_u_0, [-1, self.lstm_feat_dim]),
                                            self.w_um_1),
                                  [self.batch_size, self.class_num, self.hidden_dim_u]) )  ## batch_size, hidden_dim_v

        h_u_1 = tf.multiply(tf.tile(tf.expand_dims(WU, 0), [self.batch_size, 1, 1, 1]),
                            tf.tile(tf.expand_dims(WM, 2), [1, 1, self.phrs_Num, 1]))  # element-wise mul
        MU = tf.reshape(tf.matmul(tf.reshape(h_u_1, [-1, self.hidden_dim_u]), self.W_uh_1),
                        [self.batch_size, self.class_num, self.phrs_Num, 1])
        alpha_u = tf.nn.softmax(MU, dim=2)
        weighted_U = tf.batch_matmul(tf.transpose( tf.tile( tf.expand_dims(lstm_feat, 0), [self.batch_size, 1, 1, 1]),
                                                   [0, 1, 3, 2]),
                                     alpha_u)
        weighted_U_final = tf.tanh(tf.reshape(weighted_U, [self.batch_size,  self.class_num, self.lstm_feat_dim]) )

        return weighted_U_final

    def get_sim(self, weighted_V, weighted_U):

        return tf.reshape( tf.batch_matmul(tf.expand_dims(weighted_V, dim=1),
                                           tf.expand_dims(weighted_U, dim=2)),
                           [self.batch_size, 1])

    def get_loss(self, pred_score, groundtruth):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_score, groundtruth, name=None)

        return tf.reduce_mean(cross_entropy)

    def model(self):
        image_feat    = tf.placeholder("float32", [self.batch_size, self.parts_Num, self.feature_dim])
        phrase_feat   = tf.placeholder("int32",   [self.class_num, self.phrs_Num,  self.n_lstm_steps])
        labels        = tf.placeholder("int32",   [self.batch_size])

        self.batch_cls_Num = self.class_num #tf.shape(phrase_feat)[0]
        self.batch_phrs_size = self.batch_cls_Num * self.phrs_Num

        image_feat_mean = tf.reduce_mean(image_feat, axis=1)
        V_0 = tf.tanh( tf.matmul(image_feat_mean, self.P_0) )

        lstm_feat_2d = self.phrase_embed(tf.reshape(phrase_feat, [-1, self.n_lstm_steps]))
        lstm_feat = tf.reshape(lstm_feat_2d, [self.class_num, self.phrs_Num, self.lstm_feat_dim])
        U_0 = tf.reduce_mean(lstm_feat, axis=1)

        ## m_0 [batch_size, class_num, n_lstm_steps]

        m_0  = tf.multiply( tf.tile( tf.expand_dims(V_0, dim=1), [1, self.class_num, 1 ] ),
                            tf.tile( tf.expand_dims(U_0, dim=0), [self.batch_size, 1, 1] ) )

        weighted_V     = self.get_visual_att(image_feat, m_0) # batch_size, cls_Num, 1024


        weighted_U     = self.get_text_att(lstm_feat_2d, lstm_feat, m_0) # batch_size, cls_Num, 1024

        sim_VU = tf.reshape(tf.batch_matmul(tf.expand_dims(weighted_V, dim=2),
                                         tf.expand_dims(weighted_U, dim=3)),
                         [self.batch_size, self.class_num])

        # add regularization loss, including LSTM
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * self.l2_weight
        loss = tf.add(self.get_loss(sim_VU, labels) , lossL2)

        pred_lb = tf.cast(tf.argmax(sim_VU, axis=1), tf.int32)
        acc = tf.reduce_mean(tf.to_float(tf.equal( pred_lb, labels)))

        return image_feat, phrase_feat, labels, loss, acc, pred_lb


    def trainer(self):
        image_feat, phrase_feat, labels, loss, acc, pred_lb = self.model()

        return image_feat, phrase_feat, labels, loss, acc

    def solver(self):
        image_feat, phrase_feat, labels, loss, acc, pred_lb = self.model()

        return image_feat, phrase_feat, labels, pred_lb

if __name__ == '__main__':
    model = attention_model(128, [2048], 16000, 20000, 300, 3000, 50)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()