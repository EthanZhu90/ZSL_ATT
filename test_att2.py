import tensorflow as tf
import os
import itertools
from time import gmtime, strftime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils import *


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(config = Config('baseline')): # baseline
    epoch = 8
    l2_weight = 0.01

    #if L2 reg doesn't include LSTM, then remove 'includeLSTM'
    logFile = open( os.path.join('model', config.config_name, 'l2Weight' + str(l2_weight) + 'includeLSTM',
                                  config.config_name + '_Log.txt'), 'a')

    logFile.write(strftime("\n\n\n%Y-%m-%d %H:%M:%S\n", gmtime()))
    logFile.write('Test on ' + config.config_name + '    Epoch: %d\n\n'%(epoch))


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
    print('Totally # of phrase: %d' % (phrase_stack.shape[0]))
    logFile.write('Totally # of phrase: %d\n' % (phrase_stack.shape[0]))

    ## phrase_test_mat
    phrase_test_mat = []
    for idx in range(config.class_num_test):
        class_phrases = phrases_test[idx]
        if (class_phrases.shape[0] > 100):
            phrase_test_mat.append(class_phrases[0:100, :])
        elif (class_phrases.shape[0] < 100):
            phrase_test_mat.append(np.concatenate([class_phrases,
                                                    np.zeros([100 - class_phrases.shape[0], class_phrases.shape[1]],
                                                             dtype=np.int32)],
                                                   axis=0))
        else:
            phrase_test_mat.append(class_phrases)
    phrase_test_mat = np.array(phrase_test_mat)



    if (config.config_name != 'baseline3'):
        phrase_stack = phrase_stack.astype(int)
        word_num = np.amax(np.vstack(phrase))+ 1
    else:
        word_num = 0 ## dummy number

    model = config.model(config.batch_size, 512 * 7, 200, config.n_lstm_steps, word_num, phrs_size_all, phrs_size_stepByCls)
    model.l2_weight = l2_weight

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)

    model_subpath = os.path.join(config.model_path, 'l2Weight' + str(l2_weight) + 'includeLSTM')
    saver.restore(sess, os.path.join(model_subpath, 'model-%d' % (epoch)))
    if (config.config_name == 'coattention_model'):
        image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, weighted_V_op, weighted_U_op = model.solver()
    else:
        image_feat, phrase_feat, labels, pred_op = model.solver()

    from_idx = range(0, config.testing_num, config.batch_size)
    to_idx = range(config.batch_size, config.testing_num+config.batch_size, config.batch_size)

    pred_lb_all = np.empty((0), dtype=np.int32)
    weighted_V_all  = np.empty((0, 1024), dtype=np.float32)

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

        if (config.config_name == 'coattention_model'):
            curr_image_feat_t3     = np.reshape(curr_image_feat, [curr_image_feat.shape[0], 7, 512])
            curr_image_feat_neg_t3 = np.reshape(curr_image_feat, [curr_image_feat.shape[0], 7, 512])
            curr_phrs_feat     = np.zeros([config.batch_size, 100, config.n_lstm_steps], dtype=np.int32)
            curr_phrs_feat_neg = np.zeros([config.batch_size, 100, config.n_lstm_steps], dtype=np.int32)

            weighted_V, weighted_U = sess.run(
                [weighted_V_op, weighted_U_op],
                feed_dict={image_feat: curr_image_feat_t3,
                           image_feat_neg: curr_image_feat_neg_t3,
                           phrase_feat: curr_phrs_feat,
                           phrase_feat_neg: curr_phrs_feat_neg
                           })
            weighted_V_all = np.concatenate((weighted_V_all, weighted_V), axis=0)

        else:
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

    #### to computer accuracy and analysis.
    if (config.config_name != 'coattention_model'):
            pred_lb_all = pred_lb_all[0:lb_test.shape[0]]
            acc = float(np.sum(pred_lb_all == lb_test)) / lb_test.shape[0] * 100
            print('Accurray is %f' % acc)
            logFile.write('Accurray is %f\n' % acc)
            logFile.close()

            cnf_matrix = confusion_matrix(lb_test, pred_lb_all)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix,  classes= config.class_name_test,
                                  title='Confusion matrix, without normalization')
            plt.show()
    else:
        weighted_V_all = weighted_V_all[0:lb_test.shape[0]]
        from_idx = range(0, config.class_num_test, config.batch_size)
        to_idx = range(config.batch_size, config.class_num_test + config.batch_size, config.batch_size)

        weighted_U_all = np.empty((0, 1024), dtype=np.float32)
        for (start, end) in zip(from_idx, to_idx):
            curr_phrs_feat = phrase_test_mat[start:end, :]
            if (curr_phrs_feat.shape[0] < config.batch_size):  # for the last iteration
                curr_phrs_feat = np.concatenate((curr_phrs_feat,
                                                 np.zeros((config.batch_size - curr_phrs_feat.shape[0],
                                                            curr_phrs_feat.shape[1],
                                                            curr_phrs_feat.shape[2]))),
                                                 axis=0)
            curr_phrs_feat_neg      = curr_phrs_feat
            curr_image_feat_t3      = np.zeros([config.batch_size, 7, 512], dtype=np.float32)
            curr_image_feat_neg_t3  = np.zeros([config.batch_size, 7, 512], dtype=np.float32)

            weighted_V, weighted_U = sess.run(
                [weighted_V_op, weighted_U_op],
                feed_dict={image_feat: curr_image_feat_t3,
                           image_feat_neg: curr_image_feat_neg_t3,
                           phrase_feat: curr_phrs_feat,
                           phrase_feat_neg: curr_phrs_feat_neg
                           })
            weighted_U_all = np.concatenate((weighted_U_all, weighted_U), axis=0)
        weighted_U_all = weighted_U_all[0: config.class_num_test]
        score_mat = np.matmul( weighted_V_all, weighted_U_all.transpose())
        pred_lb_all = np.argmax( score_mat, axis=1)
        acc = float(np.sum(pred_lb_all == lb_test)) / lb_test.shape[0] * 100
        print('Accurray is %f' % acc)
        logFile.write('Accurray is %f\n' % acc)
        logFile.close()

        cnf_matrix = confusion_matrix(lb_test, pred_lb_all)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix,  classes= config.class_name_test,
                              title='Confusion matrix, without normalization')
        plt.show()

if __name__ == '__main__':
    test()