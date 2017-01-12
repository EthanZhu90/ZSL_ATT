import json, pickle, operator, re, time, os
import tensorflow as tf
import numpy as np
from config import Config
from utils import data_load, data_load3

def train(config = Config('baseline3')):

    learning_rate = 0.001
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

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
    if(config.config_name != 'baseline3'):
        phrase_stack = phrase_stack.astype(int)
    word_num = np.amax(np.vstack(phrase)) + 1

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    model = config.model(32, 512*7, 200, 4, word_num, phrs_size_all, phrs_size_stepByCls)
    #batch_size, feature_dim, embed_dim, n_lstm_steps)

    image_feat, phrase_feat, labels, predScore_op, loss_op, acc_op = model.trainer()
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

    print("*** Training Start ***")
    sum_step = 0
    iter_count = 0
    idvParser = True

    for epoch in range(config.max_epoch):
        print("Start running epoch %d" % (epoch))
        t = time.time()
        shuffler = np.random.permutation(config.training_num)
        feats_train_shuffled = feats_train[shuffler,:]
        lb_train_shuffled = lb_train[shuffler]
        for (start, end) in zip(from_idx, to_idx):
            curr_image_feat = feats_train_shuffled[start:end, :]
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

            _, pred_score, loss, acc = sess.run(
                [train_op, predScore_op, loss_op, acc_op],
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
                print("Iteration: %d\t\tloss %.6f\t\tAcc %.4f"%(iter_count, loss, acc*100))

        print("End running epoch %d : %dmin\n" % (epoch, (time.time() - t) / 60))
        saver.save(sess, os.path.join(config.model_path, 'model'),
                   global_step=epoch)


if __name__ == '__main__':
    train()