import tensorflow as tf
import json
from utils import *


def test(config = Config('baseline3')):
    epoch  = 26

    if (config.config_name == 'baseline3'):
        feats_train, lb_train, feats_test, lb_test, phrases_train, phrases_test, phrase = data_load3(config)
    else:
        feats_train, lb_train, feats_test, lb_test, phrases_train, phrases_test, phrase = data_load(config)
    ##feats_test, lb_test, feats_train, lb_train, phrases_test, phrases_train, phrases= data_load(config)

    config.testing_num = feats_test.shape[0]

    sess = tf.Session()

    phrs_size_percls = list(map(lambda x: x.shape[0], phrases_test))
    phrs_size_stepByCls = [0]
    for i in range(len(phrs_size_percls)):
        phrs_size_stepByCls.append(phrs_size_stepByCls[-1] + phrs_size_percls[i])

    phrs_size_all = np.sum(np.array(phrs_size_percls))
    phrase_stack = np.vstack(phrases_test)

    if (config.config_name != 'baseline3'):
        phrase_stack = phrase_stack.astype(int)
        word_num = np.amax(np.vstack(phrases))+ 1
    else:
        word_num = 0 ## dummy number

    model = config.model(32, 512 * 7, 200, 4, word_num, phrs_size_all, phrs_size_stepByCls)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess, config.model_path + 'model-%d' % (epoch))

    image_feat, phrase_feat, labels, pred_op = model.solver()

    from_idx = range(0, config.testing_num, config.batch_size)
    to_idx = range(config.batch_size, config.testing_num+config.batch_size, config.batch_size)

    pred_lb_all = np.empty((0), dtype=int)
    for (start, end) in zip(from_idx, to_idx):

        curr_image_feat = feats_test[start:end,:]
        curr_lb = lb_test[start:end]
        if(curr_image_feat.shape[0] < config.batch_size): # for the last iteration
            curr_image_feat = np.concatenate((curr_image_feat,
                                              np.zeros((config.batch_size - curr_image_feat.shape[0],
                                                        curr_image_feat.shape[1]))), axis=0)
            curr_lb = np.concatenate((curr_lb, np.zeros(config.batch_size - curr_lb.shape[0])))

        if (config.config_name == 'baseline2'):
            curr_lb = np.zeros([config.batch_size, phrs_size_all]) ## dummy label

        pred_lb = sess.run(
            pred_op,
            feed_dict={image_feat: curr_image_feat,
                       phrase_feat: phrase_stack,
                       labels: curr_lb
                       })
        if (config.config_name == 'baseline2'):
            avg_score = np.zeros([config.batch_size, config.class_num_test])
            index = 0
            tmp = zip(phrs_size_stepByCls[0:-1], phrs_size_stepByCls[1:])
            for start, end in zip(phrs_size_stepByCls[0:-1], phrs_size_stepByCls[1:]):
                avg_score[:, index] = np.mean(pred_lb[:, start:end], axis=1)
                index = index + 1
            pred_lb = np.argmax(avg_score, axis=1)

        pred_lb_all = np.concatenate((pred_lb_all, pred_lb), axis=0)

    pred_lb_all = pred_lb_all[0:lb_test.shape[0]]
    acc = float(np.sum(pred_lb_all == lb_test))/lb_test.shape[0] * 100
    print('Accurray is %f' % acc)
    a = 1


if __name__ == '__main__':
    test()