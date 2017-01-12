import tensorflow as tf
#import sys
#sys.path.append("/home/ethan/research/MCB_VQA/model/")
#from vqamodel import *


class attention_model():
    def __init__(self, batch_size, feature_dim, proj_dim, \
            word_num, embed_dim, ans_candi_num, n_lstm_steps):

        self.debug_dic = {}

    def model(self):
        image_feat = tf.placeholder("float32", [self.batch_size, self.feature_dim[0]])
        phrase_feat = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])


        feat = tf.concat(1, [image_feat, phrase_feat])
        normalized_feat = tf.nn.l2_normalize(feat, 0) ## do I need to add this?
        logit = tf.matmul(normalized_feat, self.fc_W) + self.fc_b

        self.debug_dic['feat'] = feat
        self.debug_dic['normal_feat'] = normalized_feat
        self.debug_dic['logit'] = logit

        return image_feat, phrase_feat, logit


    def trainer(self):
        image_feat, question, answer, logit = self.model()
        loss = self.get_loss(logit, answer)

        return image_feat, question, answer, loss

    def solver(self):
        image_feat, question, answer, logit = self.model()
        max_prob_words = tf.argmax(logit, 1)

        return image_feat, question, max_prob_words

if __name__ == '__main__':
    model = attention_model(128, [2048], 16000, 20000, 300, 3000, 50)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()