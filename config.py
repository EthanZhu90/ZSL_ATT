#from model.baseline import baseline_model
#from model.attmodel import attention_model
import sys
sys.path.append("/home/ethan/research/ZSL_ATT/model/")
from baseline import baseline_model
from baseline2 import baseline_model2
from baseline3 import baseline_model3

batch_size = 32
worddic_path = '/home/ethan/research/ZSL_ATT/data/Columbia/'
base_dir = '/home/ethan/research/ZSL_ATT/'
feats_dir = '/home/ethan/research/ZSL_ATT/data/CUB2011/cnn_feat_7part_DET_ReLU.mat'
#feats_dir = '/home/ethan/research/ZSL_ATT/data/CUB2011/cnn_feat-imagenet-vgg-verydeep-19.mat'
imgLabel_dir = '/home/ethan/research/ZSL_ATT/data/CUB2011/image_class_labels.mat'
train_test_split_dir = '/home/ethan/research/ZSL_ATT/data/CUB2011/train_test_split_easy.mat'
model_path = '/home/ethan/research/ZSL_ATT/trained_model/'
n_lstm_steps = 4

config_set = {
        'baseline' : {
            'feature_dim' : [512*7],
            'model' : baseline_model,
                      },
        'baseline2' : {
            'feature_dim' : [512*7],
            'model' : baseline_model2,
                      },
        'baseline3' : {
            'feature_dim' : [512*7],
            'model' : baseline_model3,
                      }
    }

	    #'attmodel' : {'feature_dim' : [2048],
        #    'feats_path' : feats_path,
        #    'val_feats_path' : val_feats_path,
        #    'model' : attention_model,
	    #    'proj_dim' : proj_dim}

class Config(object):
    def __init__(self, config_name='baseline'):
        self.config_name = config_name
        self.batch_size = batch_size
        attset = config_set[self.config_name]
        self.model = attset['model']
        self.training_num = 8855   ## image number
        self.model_path = base_dir+'model/'+config_name+'/'
        self.checkpoint = None
        self.max_epoch = 30
        self.feats_dir = feats_dir
        self.imgLabel_dir = imgLabel_dir
        self.train_test_split_dir = train_test_split_dir
        self.worddic_path = worddic_path
        self.n_lstm_steps = n_lstm_steps
        self.class_num = 150
        self.class_num_test = 50

if __name__ == '__main__':
    config = Config()
