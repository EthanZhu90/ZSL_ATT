import numpy as np
import scipy.io as sio
import json
import h5py

from config import Config
from keras.preprocessing import sequence


def data_load(config):

    mat = sio.loadmat(config.feats_dir)
    feats = np.array(mat['cnn_feat'])
    feats = feats.transpose();
    mat = sio.loadmat(config.imgLabel_dir)
    imgLabel = np.array(mat['imageClassLabels'])
    mat = sio.loadmat(config.train_test_split_dir)
    train_cid = np.array(mat['train_cid'])
    test_cid = np.array(mat['test_cid'])

    # (1) prepare training data
    train_img = np.empty((0,2), dtype=int)
    for i in list(train_cid.flatten()):
        img_t = imgLabel[imgLabel[:,1] == i ]
        train_img = np.append(train_img, img_t, axis=0)

    tmp_train_flag = np.zeros(imgLabel.shape[0])
    for i in train_img[:,0]:
        tmp_train_flag[i-1] = 1
    feats_train = feats[tmp_train_flag == 1, :]
    tmp_train_flag = None

    train_lb_true = train_img[:,1]
    train_lb_uq = np.unique(train_lb_true)
    #train_lb_t2f, train_lb_f2t = {}, {}

    train_lb_fake = np.copy(train_lb_true)
    for i in range(len(train_lb_uq)):
        train_lb_fake[train_lb_true == train_lb_uq[i]] = i

    # (2) prepare testing  data

    test_img = np.empty((0, 2), dtype=int)
    for i in list(test_cid.flatten()):
        img_t = imgLabel[imgLabel[:, 1] == i]
        test_img = np.append(test_img, img_t, axis=0)

    tmp_test_flag = np.zeros(imgLabel.shape[0])
    for i in test_img[:, 0]:
        tmp_test_flag[i - 1] = 1
    feats_test = feats[tmp_test_flag == 1, :]
    tmp_test_flag = None

    test_lb_true = test_img[:, 1]
    test_lb_uq = np.unique(test_lb_true)
    # train_lb_t2f, train_lb_f2t = {}, {}

    test_lb_fake = np.copy(test_lb_true)
    for i in range(len(test_lb_uq)):
        test_lb_fake[test_lb_true == test_lb_uq[i]] = i

    # (3) prepare training phrases

    p_word2ix = json.load(open(config.worddic_path + 'p_word2ix.json', 'rb'))
    #p = re.compile('[\w]+')
    class_phrase_path = 'data/Columbia/class_phrases.json'
    class_phrases = json.load(open(class_phrase_path, 'rb'))
    phrases = []
    for i in class_phrases:
        phrase_per_class = list(map(lambda phrs:
                             [p_word2ix[word] for word in phrs.lower().split()
                              if word in p_word2ix],
                             i))
        phrase_per_class = np.array(sequence.pad_sequences(
            phrase_per_class, padding='post', maxlen=config.n_lstm_steps))
        phrases.append(phrase_per_class)
    train_phrases = []
    test_phrases = []
    for i in range(len(phrases)):
        if i+1 in list(train_cid.flatten()):
            train_phrases.append(phrases[i])
        if i+1 in list(test_cid.flatten()):
            test_phrases.append(phrases[i])

    return feats_train, train_lb_fake, feats_test, test_lb_fake, train_phrases, test_phrases, phrases

# this is for baseline3
def data_load3(config):
    myFile = h5py.File('/home/ethan/research/SentEmbed_reed/fea_txt.h5', 'r')
    phrase_embed_all = myFile['data'][...]


    mat = sio.loadmat(config.feats_dir)
    feats = np.array(mat['cnn_feat'])
    feats = feats.transpose();
    mat = sio.loadmat(config.imgLabel_dir)
    imgLabel = np.array(mat['imageClassLabels'])
    mat = sio.loadmat(config.train_test_split_dir)
    train_cid = np.array(mat['train_cid'])
    test_cid = np.array(mat['test_cid'])

    # (1) prepare training data
    train_img = np.empty((0,2), dtype=int)
    for i in list(train_cid.flatten()):
        img_t = imgLabel[imgLabel[:,1] == i ]
        train_img = np.append(train_img, img_t, axis=0)

    tmp_train_flag = np.zeros(imgLabel.shape[0])
    for i in train_img[:,0]:
        tmp_train_flag[i-1] = 1
    feats_train = feats[tmp_train_flag == 1, :]
    tmp_train_flag = None

    train_lb_true = train_img[:,1]
    train_lb_uq = np.unique(train_lb_true)
    #train_lb_t2f, train_lb_f2t = {}, {}

    train_lb_fake = np.copy(train_lb_true)
    for i in range(len(train_lb_uq)):
        train_lb_fake[train_lb_true == train_lb_uq[i]] = i

    # (2) prepare testing  data

    test_img = np.empty((0, 2), dtype=int)
    for i in list(test_cid.flatten()):
        img_t = imgLabel[imgLabel[:, 1] == i]
        test_img = np.append(test_img, img_t, axis=0)

    tmp_test_flag = np.zeros(imgLabel.shape[0])
    for i in test_img[:, 0]:
        tmp_test_flag[i - 1] = 1
    feats_test = feats[tmp_test_flag == 1, :]
    tmp_test_flag = None

    test_lb_true = test_img[:, 1]
    test_lb_uq = np.unique(test_lb_true)
    # train_lb_t2f, train_lb_f2t = {}, {}

    test_lb_fake = np.copy(test_lb_true)
    for i in range(len(test_lb_uq)):
        test_lb_fake[test_lb_true == test_lb_uq[i]] = i

    # (3) prepare training phrases feature ( extracted by reet )

    p_word2ix = json.load(open(config.worddic_path + 'p_word2ix.json', 'rb'))
    #p = re.compile('[\w]+')
    class_phrase_path = 'data/Columbia/class_phrases.json'
    class_phrases = json.load(open(class_phrase_path, 'rb'))
    #phrases = []
    phrase_embed_all_tmp = []
    idx = 0
    for i in class_phrases:
        phrase_per_class = list(map(lambda phrs:
                             [p_word2ix[word] for word in phrs.lower().split()
                              if word in p_word2ix],
                             i))
        phrase_embed_all_tmp.append(phrase_embed_all[idx: idx+ len(phrase_per_class)])
        idx = idx+ len(phrase_per_class)

        #phrases.append()
        #
        #phrase_per_class = np.array(sequence.pad_sequences(
        #    phrase_per_class, padding='post', maxlen=config.n_lstm_steps))
        #phrases.append(phrase_per_class)
    train_phrases = []
    test_phrases = []
    for i in range(len(phrase_embed_all_tmp)):
        if i+1 in list(train_cid.flatten()):
            train_phrases.append(phrase_embed_all_tmp[i])
        if i+1 in list(test_cid.flatten()):
            test_phrases.append(phrase_embed_all_tmp[i])

    return feats_train, train_lb_fake, feats_test, test_lb_fake, train_phrases, test_phrases, phrase_embed_all_tmp
