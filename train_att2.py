import json, pickle, operator, re, time, os, sys
import tensorflow as tf
import numpy as np
from time import gmtime, strftime

from config import Config
from utils import data_load, data_load3

def get_neg_img_sample( feats_train, lb_train, lb_train_shuffled, config):
    ####  get the neg image sample

    neg_random_samples = []
    for class_idx in range(config.class_num):
        shuffler_neg = np.random.permutation(config.training_num)
        neg_random_samples.append(list(filter(lambda x: lb_train[x] != class_idx, shuffler_neg)))

    feats_train_neg = []
    lb_train_neg = []
    sentinel = np.zeros(config.class_num).astype(int)

    for idx in range(lb_train_shuffled.shape[0]):
        lb_pos = lb_train_shuffled[idx]
        feats_train_neg.append(feats_train[neg_random_samples[lb_pos][sentinel[lb_pos]]])
        lb_train_neg.append(lb_train[neg_random_samples[lb_pos][sentinel[lb_pos]]])
        sentinel[lb_pos] += 1
    return np.array(feats_train_neg)

def get_neg_txt_sample(phrase_trainMat_shuffled, lb_train_shuffled, config):
    ####  get the neg textual  sample
    neg_random_samples = []  ## range [0 149)
    for class_idx in range(config.class_num):
        shuffler_neg = np.floor(np.random.random_sample((config.training_num,)) * config.class_num).astype(np.int32)
        neg_random_samples.append(list(filter(lambda x: x != class_idx, shuffler_neg)))

    phrase_train_neg = []
    phrase_lb_train_neg = []
    sentinel = np.zeros(config.class_num).astype(np.int32)

    for idx in range(lb_train_shuffled.shape[0]):
        lb_pos = lb_train_shuffled[idx]
        phrase_train_neg.append(phrase_trainMat_shuffled[neg_random_samples[lb_pos][sentinel[lb_pos]]])
        phrase_lb_train_neg.append(neg_random_samples[lb_pos][sentinel[lb_pos]])
        sentinel[lb_pos] += 1
    return np.array(phrase_train_neg)

def phrase2classLabel(phraseLabel,  phrs_size_stepByCls):
    classLabel = []
    for label in phraseLabel:
        for j in range(len(phrs_size_stepByCls)):
            if(label<phrs_size_stepByCls[j]):
                break
        classLabel.append(j-1)

    return classLabel

def train(config = Config('coattmodel2'), l2_weight=0.01 , gpuID = '0'):
    #coattention_model
    #available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    #os.environ['CUDA_VISIBLE_DEVICES'] = available_devices[gpu]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuID

    model_subpath = os.path.join(config.model_path, 'l2Weight' + str(l2_weight) +'includeLSTM')
    if not os.path.exists(model_subpath):
        os.makedirs(model_subpath)

    learning_rate = 0.001

    logFile = open( os.path.join(model_subpath, config.config_name + '_Log.txt'), 'a')
    logFile.write(strftime("\n\n\n%Y-%m-%d %H:%M:%S\n", gmtime()))
    logFile.write('Experiment on '+ config.config_name + '\n\n')

    if(config.config_name == 'baseline3'):
        feats_train, lb_train, feats_test, lb_test, phrases_train, phrases_test, phrase = data_load3(config)
    else:
        feats_train, lb_train, feats_test, lb_test, phrases_train, phrases_test, phrase = data_load(config)


    config.training_num = feats_train.shape[0]

    phrs_size_percls = list(map(lambda x: x.shape[0], phrases_train))
    phrs_size_stepByCls = [0]
    for i in range(len(phrs_size_percls)):
        phrs_size_stepByCls.append(phrs_size_stepByCls[-1] + phrs_size_percls[i])

    phrs_size_all = np.sum(np.array(phrs_size_percls))
    phrase_stack = np.vstack(phrases_train)
    print('Totally # of phrase: %d'%(phrase_stack.shape[0]))

    ## phrase_train_mat
    phrase_train_mat = []
    for idx in range(config.class_num):
        class_phrases = phrases_train[idx]
        if(class_phrases.shape[0] > 100):
            phrase_train_mat.append(class_phrases[0:100, :])
        elif (class_phrases.shape[0] < 100):
            phrase_train_mat.append(np.concatenate( [class_phrases,
                                                     np.zeros([100 - class_phrases.shape[0],class_phrases.shape[1]], dtype=np.int32)],
                                                     axis=0) )
        else:
            phrase_train_mat.append(class_phrases)
    phrase_train_mat = np.array(phrase_train_mat)

    if(config.config_name != 'baseline3'):
        phrase_stack = phrase_stack.astype(int)
    word_num = np.amax(np.vstack(phrase)) + 1

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    model = config.model(config.batch_size, 512*7, 200, config.n_lstm_steps,
                         word_num, phrs_size_all, phrs_size_stepByCls)

    model.l2_weight = l2_weight

    #batch_size, feature_dim, embed_dim, n_lstm_steps)

    if (config.config_name == 'coattention_model'):
        image_feat, image_feat_neg, phrase_feat, phrase_feat_neg, loss_op, acc_op = model.trainer()
    else:
        image_feat, phrase_feat, labels, predScore_op, loss_op, loss1_op, lossL2_op, acc_op = model.trainer()
    #image_feat, phrase_feat, labels, max_prob_words = model.solver()
    saver = tf.train.Saver(max_to_keep=50)

    # Apply grad clipping
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss_op)
    clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) \
                   for grad, var in gvs if not grad is None]
    train_op = optimizer.apply_gradients(clipped_gvs)

    #sum_writer = tf.train.SummaryWriter(config.log_path, sess.graph)
    #loss_sum_op = tf.scalar_summary('train_loss', loss_op)

    sess.run(tf.global_variables_initializer())
    #if config.checkpoint:
    #    saver.restore(sess, config.model_path + "-%d" % (checkpoint))

    from_idx = range(0, config.training_num, config.batch_size)
    to_idx = range(config.batch_size, config.training_num, config.batch_size)


    print('Model: %s   l2Weight: %s' % (config.config_name,  str(l2_weight)))
    print("*** Training Start ***")
    sum_step = 0
    iter_count = 0
    idvParser = True

    #
    for epoch in range(config.max_epoch):
        print("Start running epoch %d" % (epoch))
        logFile.write("Start running epoch %d\n" % (epoch))

        t = time.time()
        shuffler = np.random.permutation(config.training_num)
        feats_train_shuffled = feats_train[shuffler,:]
        lb_train_shuffled = lb_train[shuffler]


        ####  get the neg image sample
        feats_train_neg = get_neg_img_sample(feats_train, lb_train, lb_train_shuffled, config)

        ###  get the pos textual sample
        phrase_trainMat_shuffled = phrase_train_mat[lb_train_shuffled, :]

        ###  get the neg textual sample
        phrase_train_neg = get_neg_txt_sample(phrase_trainMat_shuffled, lb_train_shuffled, config)




        for (start, end) in zip(from_idx, to_idx):
            curr_image_feat = feats_train_shuffled[start:end, :]
            curr_image_feat_neg = feats_train_neg[start:end, :]
            curr_phrs_feat  = phrase_trainMat_shuffled[start:end, :]
            curr_phrs_feat_neg = phrase_train_neg[start:end, :]

            curr_lb = lb_train_shuffled[start:end]
            curr_lb_org = curr_lb
            #gt_matrix = np.zeros(config.batch_size, config.class_num)
            #for i in range(len(curr_lb)):
            #    gt_matrix[i, curr_lb[i]] = 1
            if(config.config_name == 'baseline2'):
                label_mat = np.zeros([config.batch_size, phrs_size_all])
                for i in range(config.batch_size):
                    start = phrs_size_stepByCls[curr_lb[i]]
                    end = phrs_size_stepByCls[curr_lb[i] + 1]
                    label_mat[i, start:end] = 1

                curr_lb = label_mat
            if(config.config_name == 'coattention_model'):
                curr_image_feat_t3  = np.reshape(curr_image_feat, [curr_image_feat.shape[0], 7, 512])
                curr_image_feat_neg_t3 = np.reshape(curr_image_feat_neg, [curr_image_feat.shape[0], 7, 512])


                _, loss, acc = sess.run(
                    [train_op, loss_op, acc_op],
                    feed_dict={image_feat: curr_image_feat_t3,
                               image_feat_neg: curr_image_feat_neg_t3,
                               phrase_feat: curr_phrs_feat,
                               phrase_feat_neg: curr_phrs_feat_neg
                               })
            else:
                _, pred_score, loss, loss1, lossL2,  acc = sess.run(
                    [train_op, predScore_op, loss_op, loss1_op, lossL2_op, acc_op],
                    feed_dict={image_feat: curr_image_feat,
                               phrase_feat: phrase_stack,
                               labels: curr_lb
                               })


            iter_count += 1
            if iter_count % 50 == 0:
                if (config.config_name == 'baseline2'):
                    avg_score = np.zeros([config.batch_size, config.class_num])
                    index = 0
                    tmp = zip(phrs_size_stepByCls[0:-1], phrs_size_stepByCls[1:])
                    for start, end in zip(phrs_size_stepByCls[0:-1], phrs_size_stepByCls[1:]):
                        avg_score[:, index] = np.mean(pred_score[:, start:end], axis=1)
                        index = index + 1
                    max_prob_words = np.argmax(avg_score, axis=1)
                    acc = np.mean(np.equal(max_prob_words, curr_lb_org))

                    ## the second accuracy
                    phraseLabel = np.argmax(pred_score, axis=1)
                    max_prob_words = phrase2classLabel(phraseLabel,  phrs_size_stepByCls)
                    acc2 = np.mean(np.equal(max_prob_words, curr_lb_org))
                    print("Iteration: %d\t\tloss %.6f\t\tAcc %.4f\t\tAcc2 %.4f" % (
                    iter_count, loss, acc * 100, acc2 * 100))
                    logFile.write("Iteration: %d\t\tloss %.6f\t\tAcc %.4f\t\tAcc2 %.4f\n" % (
                    iter_count, loss, acc * 100, acc2 * 100))
                elif(config.config_name == 'coattention_model'):
                    print("Iteration: %d\t\tloss %.6f\t\tAcc %.4f" %
                          (iter_count, loss, acc * 100))
                    logFile.write("Iteration: %d\t\tloss %.6f\t\tAcc %.4f\n" %
                          (iter_count, loss, acc * 100))
                else:
                    print("Iteration: %d\t\tloss %.6f\t\tloss1 %.6f\t\tlossL2 %.6f\t\tAcc %.4f"%
                          (iter_count, loss, loss1, lossL2, acc*100))
                    logFile.write("Iteration: %d\t\tloss %.6f\t\tloss1 %.6f\t\tlossL2 %.6f\t\tAcc %.4f\n"%
                                  (iter_count, loss, loss1, lossL2, acc*100))

        print("End running epoch %d : %dmin\n" % (epoch, (time.time() - t) / 60))
        logFile.write("End running epoch %d : %dmin\n" % (epoch, (time.time() - t) / 60))
        if epoch % 2 == 0:
            saver.save(sess, os.path.join(model_subpath, 'model'),
                   global_step=epoch)
    logFile.close()



if __name__ == '__main__':
    if(len(sys.argv) == 1):
        train()
    else:
        gpuID = sys.argv[1][3]
        modelName = sys.argv[2]
        l2Weight  = float(sys.argv[3])
        train(Config(modelName), l2Weight, gpuID)
