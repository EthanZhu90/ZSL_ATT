import json, operator, re, time, pickle, os
import numpy as np
#from IPython import embed
#from cnn import *

training_img_num = 5000
validation_img_num = 500
word_num = 1000
ans_candi_num = 1000

# Caffe model : ResNet
res_model = '/home/ethan/caffe/models/resnet/ResNet-101-model.caffemodel'
res_deploy ='/home/ethan/caffe/models/resnet/ResNet-101-deploy.prototxt'

layer_set = {
        '4b' : {'layers' : 'res4b22_branch2c', 'layer_size' : [1024, 14, 14], 'feat_path' : '/home/ethan/research/MCB_VQA/data/features/train_res4b_feat.npy'},
        'default' : {'layers' : 'pool5', 'layer_size' : [2048], 'feat_path' : '/home/ethan/research/MCB_VQA/data/features/train_res_feat.npy'}
        }

#annotations_path = '/data1/shmsw25/vqa/mscoco_train2014_annotations.json'
#questions_path = '/data1/shmsw25/vqa/OpenEnded_mscoco_train2014_questions.json'

#annotations_result_path = '/home/ethan/research/MCB_VQA/data/train_annotations_result'
#worddic_path = '/home/ethan/research/MCB_VQA/data/'

#image_path = '/home/ethan/research/im2txt/im2txt/data/raw-data/train2014/COCO_train2014_'
#image_path = '/data1/common_datasets/mscoco/images/train2014/COCO_train2014_'
#imgix2featix_path = '/home/ethan/research/MCB_VQA/data/img2feat'
class_phrase_path = 'data/Columbia/class_phrases.json'
worddic_path = 'data/Columbia/'

def create_annotations_result():


    #annotations = json.load(open(annotations_path, 'rb'))['annotations'][:3*training_img_num]
    #questions = json.load(open(questions_path, 'rb'))['questions'][:3*training_img_num]

    class_phrases = json.load(open(class_phrase_path, 'rb'))
    p_dic, p_word2ix, p_ix2word = {}, {}, {}

    for phrases in class_phrases:
        for phr in phrases:
            words = phr.split()
            for word in words:
                if word in p_dic: p_dic[word] += 1
                else: p_dic[word] = 1

    p_dic = sorted(p_dic.items(), key=operator.itemgetter(1))
    p_dic.reverse()
    print('The number of words in answers is %d.'%len(p_dic) )
    print("Top 20 frequent answers : ")
    for i in range(20):
        print(p_dic[i][0], p_dic[i][1])


    for i in range(len(p_dic)):
        p_word2ix[p_dic[i][0]] = i
        p_ix2word[i] = p_dic[i][0]

    # p_dic now is list, word2ix, ix2word are dictionary
    print("Answer word2ix, ix2word created.")

    with open(worddic_path + 'p_word2ix.json', 'w') as f:
        json.dump(p_word2ix, f)
    with open(worddic_path + 'p_ix2word.json', 'w') as f:
        json.dump(p_ix2word, f)


'''
    image_id_list, question_list, answer_list = [], [], []

    q_dic, q_word2ix, q_ix2word = {}, {}, {}
    a_dic, a_word2ix, a_ix2word = {}, {}, {}

    # (1) create wordtoix, ixtoword for answers
    answer_type_dic = {}
    for dic in annotations:
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes':
                ans = a['answer']
                if ans in a_dic: a_dic[ans] += 1
                else: a_dic[ans] = 1
                ans_type = dic['answer_type']
                if ans_type in answer_type_dic:
                    answer_type_dic[ans_type] += 1
                else:
                    answer_type_dic[ans_type] = 1
    a_dic = sorted(a_dic.items(), key=operator.itemgetter(1))
    a_dic.reverse()
    #print "Answer Type Dic"
    #print answer_type_dic
    #print "The number of words in answers is %d. Select only %d words."%(len(a_dic), ans_candi_num)
    #print "Top 20 frequent answers : "
    for i in range(20):
        #print a_dic[i][0], a_dic[i][1]
    for i in range(ans_candi_num):
        a_word2ix[a_dic[i][0]] = i
        a_ix2word[i] = a_dic[i][0]
    #print "Answer word2ix, ix2word created. Threshold is %d"%(a_dic[ans_candi_num][1])

    # (2) create wordtoix, ixtoword for questions
    p = re.compile('[\w]+')
    q_len_dic, q_freq_dic = {}, {}
    for dic in questions:
        q_dic[(dic['image_id'], dic['question_id'])] = dic['question']
        q_words = p.findall(dic['question'].lower())
        for qw in q_words:
            if qw in q_freq_dic : q_freq_dic[qw] += 1
            else: q_freq_dic[qw] = 1
        q_len_key = 10*int(len(q_words)/10)
        if q_len_key in q_len_dic : q_len_dic[q_len_key] += 1
        else : q_len_dic[q_len_key] = 1
    print "Length of questions"
    for q_len_key in q_len_dic:
        print "%d ~ %d\t: %d" %(q_len_key, q_len_key+10, q_len_dic[q_len_key])
    print "Total\t: %d" %(sum(q_len_dic.values()))
    q_freq_dic = sorted(q_freq_dic.items(), key=operator.itemgetter(1))
    q_freq_dic.reverse()
    print "The number of words in questions is %d. Select only %d words."%(len(q_freq_dic), word_num)
    q_word2ix['?'] = 0
    q_ix2word[0] = '?'
    for i in range(1, word_num):
        q_word2ix[q_freq_dic[i-1][0]] = i
        q_ix2word[i] = q_freq_dic[i-1][0]
    print "Question word2ix, ix2word created. Threshold is %d"%(q_freq_dic[word_num][1])

    # (3) create annotations_result
    num = 0
    answer_freq_dic = {}
    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        i = 0
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes' and a['answer'] in a_word2ix:
                image_id_list.append(dic['image_id'])
                question_list.append(q)
                answer_list.append(a_word2ix[a['answer']])
                i += 1
	    if i==0: num+=1
    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({'image_ids' : image_id_list,
        'questions' : question_list,
        'answers' : answer_list},
        open(annotations_result_path, 'wb'))
    print "Success to save Annotation results"

    pickle.dump(q_word2ix, open(worddic_path+'q_word2ix', 'wb'))
    pickle.dump(q_ix2word, open(worddic_path+'q_ix2word', 'wb'))
    pickle.dump(a_word2ix, open(worddic_path+'a_word2ix', 'wb'))
    pickle.dump(a_ix2word, open(worddic_path+'a_ix2word', 'wb'))
    print "Success to save Worddics"

    # (4) Create image features
    # If you run this seperatly, load image_id_list
    #image_id_list = pickle.load(open(annotations_result_path, 'rb'))['image_ids']
    unique_image_ids = list(set(image_id_list))

    unique_images = map(lambda x: \
            os.path.join(image_path+("%12s"%str(x)).replace(" ","0")+".jpg"),  #.replace(" ","0")
		unique_image_ids)
    imgix2featix = {}
    for i in range(len(unique_images)):
        imgix2featix[unique_image_ids[i]] = i
    pickle.dump(imgix2featix, open(imgix2featix_path, 'wb'))

    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)
    print len(unique_images)

    for dic in layer_set.values():
        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        if not os.path.exists(feat_path):
            feats = cnn.get_features(unique_images,
                layers=layers, layer_sizes=layer_size)
	    print feats.shape
            np.save(feat_path, feats)
    print "Success to save features"
'''
if __name__ == '__main__':
    create_annotations_result()
