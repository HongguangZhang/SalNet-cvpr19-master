import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import time
import models

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num_per_class",type = int, default = 1)
parser.add_argument("-q","--query_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 100)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-sigma","--sigma", type = float, default = 150)
parser.add_argument("-beta","--beta", type = float, default = 0.01)
args = parser.parse_args()


# Hyper Parameters
METHOD = "SalNet_IntraClass"
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.support_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma

def power_norm(x, SIGMA):
	out = 2*F.sigmoid(SIGMA*x) - 1
	return out
	
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print("init data folders")
    metatrain_folders,metaquery_folders = tg.mini_imagenet_folders()

    print("init neural networks")
    foreground_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    background_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    mixture_network = models.MixtureNetwork().apply(weights_init).cuda(GPU)
    relation_network = models.SimilarityNetwork(FEATURE_DIM,RELATION_DIM).apply(weights_init).cuda(GPU)
	
	# Loading models
    if os.path.exists(str(METHOD + "/miniImagenet_foreground_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        foreground_encoder.load_state_dict(torch.load(str(METHOD + "/miniImagenet_foreground_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load foreground encoder success")
    if os.path.exists(str(METHOD + "/miniImagenet_background_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        background_encoder.load_state_dict(torch.load(str(METHOD + "/miniImagenet_background_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load background encoder success")        
    if os.path.exists(str(METHOD + "/miniImagenet_mixture_network_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        mixture_network.load_state_dict(torch.load(str(METHOD + "/miniImagenet_mixture_network_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load mixture network success")
    if os.path.exists(str(METHOD + "/miniImagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str(METHOD + "/miniImagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    best_accuracy = 0.0
    best_h = 0.0
            
    for episode in range(EPISODE):
        with torch.no_grad():
            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metaquery_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,15)
                support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 3
                query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                support_img,support_sal,support_labels = support_dataloader.__iter__().next()
                for query_img,query_sal,query_labels in query_dataloader:
                    query_size = query_labels.shape[0]
        	    	# calculate foreground and background features
                    support_foreground = foreground_encoder(Variable(support_img*support_sal).cuda(GPU)).view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    support_background = background_encoder(Variable(support_img*(1-support_sal)).cuda(GPU)).view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    query_foreground = foreground_encoder(Variable(query_img*query_sal).cuda(GPU))
                    query_background = background_encoder(Variable(query_img*(1-query_sal)).cuda(GPU))
                    
        	    	# Intra-class Hallucination
                    support_foreground = support_foreground.unsqueeze(2).repeat(1,1,SUPPORT_NUM_PER_CLASS,1,1,1)
                    support_background = support_background.unsqueeze(1).repeat(1,SUPPORT_NUM_PER_CLASS,1,1,1,1)
                    support_mix_features = mixture_network((support_foreground + support_background).view(CLASS_NUM*(SUPPORT_NUM_PER_CLASS**2),64,19,19)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,64,19**2).sum(2).sum(1)
                    query_mix_features = mixture_network(query_foreground + query_background).view(-1,64,19**2)
                    so_support_features = Variable(torch.Tensor(CLASS_NUM, 1, 64, 64)).cuda(GPU)
                    so_query_features = Variable(torch.Tensor(query_size, 1, 64, 64)).cuda(GPU)
                    
        	    	# second-order features
                    for d in range(support_mix_features.size()[0]):
                        s = support_mix_features[d,:,:].squeeze(0)
                        s = (1.0 / support_mix_features.size()[2]) * s.mm(s.transpose(0,1))
                        so_support_features[d,:,:,:] = power_norm(s / s.trace(),SIGMA)
                    for d in range(query_mix_features.size()[0]):
                        s = query_mix_features[d,:,:].squeeze(0)
                        s = (1.0 / query_mix_features.size()[2]) * s.mm(s.transpose(0,1))
                        so_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                        
                    # calculate relations with 64x64 second-order features
                    support_features_ext = so_support_features.unsqueeze(0).repeat(query_size,1,1,1,1)
                    query_features_ext = so_query_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    query_features_ext = torch.transpose(query_features_ext,0,1)
                    relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
                    _,predict_labels = torch.max(relations.data,1)
                    rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(query_size)]
                    total_rewards += np.sum(rewards)
                    counter += query_size
                    
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)
            test_accuracy,h = mean_confidence_interval(accuracies)
            print("test accuracy:",test_accuracy,"h:",h)           
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_h = h
            print("best accuracy:",best_accuracy,"h:",best_h)
            
if __name__ == '__main__':
    main()
