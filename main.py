#coding:utf-8
import warnings
warnings.filterwarnings("ignore")
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import sqlite3
#import torchsnooper
#import task_generator as tg
import task_generator_META_DDIE as tg
import os
import math
import argparse
import random
import time
from smiles2vector import smiles2vector
from dde_config import dde_NN_config
from dde_torch import dde_NN_Large_Predictor
#10个样本的事件为175
#5个样本截止为197
CLASS_NUM=175
NUM_WAYS=5
Support_NUM_PER_CLASS=1
QUERY_NUM_PER_CLASS=4
dropoutRate=0.5
FEATURE_DIMENSION=64 #64
FLAT=2048
RELATION_DIMENSION=8
LEARNING_RATE=0.0001
EPISODE=1000000
TEST_EPISODE=5000
SMILE_SHAPE=3535
GPU=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.reluDrop = nn.Sequential(nn.ReLU(),nn.Dropout(dropoutRate))
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(32,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        #self.fc1 = nn.Sequential(nn.Linear(832,256),self.reluDrop,nn.BatchNorm1d(256)) # for 343 input dimension
        #self.fc1 = nn.Sequential(nn.Linear(5504,256),self.reluDrop,nn.BatchNorm1d(256)) # for 2161 input dimension
        self.fc1 = nn.Sequential(nn.Linear(4384, 256), self.reluDrop, nn.BatchNorm1d(256))  # for 1722 input dimension
        self.fc2 = nn.Sequential(nn.Linear(256,64),self.reluDrop,nn.BatchNorm1d(64))
        self.fc3 = nn.Sequential(nn.Linear(1024,256),self.reluDrop,nn.BatchNorm1d(256))
        self.fc4 = nn.Linear(64,1)

    def forward(self,x):
        x = torch.reshape(x,(-1,1,x.shape[1]))
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        out = F.sigmoid(self.fc4(out))
        return out

class RelationNetwork1(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork1, self).__init__()
        self.reluDrop = nn.Sequential(nn.ReLU(),nn.Dropout(dropoutRate))
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(32,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.fc1 = nn.Sequential(nn.Linear(20128,4096),self.reluDrop,nn.BatchNorm1d(4096))
        #self.fc2 = nn.Sequential(nn.Linear(4096,1024),self.reluDrop,nn.BatchNorm1d(1024))
        self.fc3 = nn.Sequential(nn.Linear(4096,256),self.reluDrop,nn.BatchNorm1d(256))
        self.fc4 = nn.Linear(256,1)


    def forward(self,x):
        #x = torch.reshape(x,(-1,1,15740))
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.sigmoid(self.fc4(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0]  * m.out_channels # m.kernel_size[1]针对Conv2D
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

import sys
class Logger(object):
  def __init__(self, filename="Default.log"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  def flush(self):
    pass



def main():
    excludeLabel=4
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(str(excludeLabel)+'.txt')
    #read data
    config = dde_NN_config()
    model_nn = dde_NN_Large_Predictor(**config)
    conn=sqlite3.connect("../METADDIEdata/Drug_META_DDIE.db")
    smile = {}
    #simFile = pd.read_csv("METADDIEdata/drug_similarity.csv")
    smileFile = pd.read_sql("select * from drug",conn)
    #simFile = pd.read_excel("METADDIEdata/drug_similarity.xlsx")
    # embedding_df = pd.read_csv("DeepDDIdata/RDF2Vec_sg_300_5_5_15_2_500_d5_uniform.txt", delimiter='\t')
    # embedding = embedding_df.loc[0:, "feature0":"feature299"]
    # embedding = np.array(embedding)
    # drug_name = embedding_df['Entity']
    # drug = []
    # for i in drug_name:
    #     drug.append(i[-8:-1])
    # drug = np.array(drug)
    #drug = np.reshape(drug, (1, 2523))

    # embedding = np.load("../METADDIEdata/drug_embed.npy")
    # drug_name = pd.read_excel("../METADDIEdata/drug_list.xlsx", header=None)
    # embedFeature = {}
    # for i in range(10551):
    #     embedFeature[drug_name.loc[i][0]]=embedding[i,:]
    for i in range(SMILE_SHAPE):
        smile[smileFile.loc[i][0]] = (smileFile.loc[i][3])

    #init neural network
    print("init neural networks")
    relation_network = RelationNetwork(20128,4096)
    #relation_network = RelationNetwork1()

    model_nn.apply(weights_init)
    model_nn=model_nn.to(device)
    relation_network.apply(weights_init)
    relation_network=relation_network.to(device)

    #feature_encoder.cuda(GPU)
    #relation_network.cuda(GPU)

    #excludeLabel = np.random.randint(5)

    model_nn_optim = torch.optim.Adam(model_nn.parameters(), lr = LEARNING_RATE)
    model_nn_scheduler = StepLR(model_nn_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    if os.path.exists(str("models1005/model_nn_" + str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl")):
        model_nn.load_state_dict(torch.load(str("models1005/model_nn_" + str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"),map_location=device))
        print("load CASTER success")
    if os.path.exists(str("models1005/relation_network_"+ str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl")):
        relation_network.load_state_dict(torch.load(str("models1005/relation_network_"+ str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"),map_location=device))
        print("load relation network success")

    #start training
    last_accuracy=0.0
    seen_acc = 0.0
    total_train_rewards = 0

    start = time.time()

    for episode in range(EPISODE):

        # test_pairs = np.reshape(smiles2vector(smile['DB01020'], smile['DB00203']), (1, -1))
        # a = smile['DB01020']
        # b = smile['DB00203']
        # test_pairs = torch.from_numpy(test_pairs).to(device)
        # a, b, c, d, e = model_nn(test_pairs.float())
        # c = c.detach().cpu().numpy()

        # print(abs(c[0, 195]), abs(c[0, 173]), abs(c[0, 149]))
        # print((abs(c[0, 195])+abs(c[0, 173])+abs(c[0, 149]))/3)


        model_nn_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        task = tg.MetaDDIETask(CLASS_NUM, NUM_WAYS, Support_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,"train",excludeLabel)
        support_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader = tg.get_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        support_drug1, support_drug2, support_labels = support_dataloader.__iter__().next()
        query_drug1, query_drug2, query_labels = query_dataloader.__iter__().next()
        support_sample_drugs = np.zeros((0,config['input_dim']))
        query_sample_drugs = np.zeros((0,config['input_dim']))
        for i in range(Support_NUM_PER_CLASS*NUM_WAYS):
            support_sample_drugs = np.vstack((support_sample_drugs,np.reshape(smiles2vector(smile[support_drug1[i]],smile[support_drug2[i]]),(1,-1))))
        for i in range(QUERY_NUM_PER_CLASS*NUM_WAYS):
            query_sample_drugs = np.vstack((query_sample_drugs,np.reshape(smiles2vector(smile[query_drug1[i]],smile[query_drug2[i]]),(1,-1))))
        #support_sample_drugs=torch.Tensor([smiles2vector(smile[x],smile[y]) for x in support_drug1 for y in support_drug2])
        #query_sample_drugs=torch.Tensor([smiles2vector(smile[x],smile[y]) for x in query_drug1 for y in query_drug2])
        support_sample_drugs = torch.from_numpy(support_sample_drugs).to(device)
        query_sample_drugs = torch.from_numpy(query_sample_drugs).to(device)
        recon1, support_feature, mag_support_feature, Z_f1, z_D1 = model_nn(support_sample_drugs.float())
        recon2, query_feature, mag_query_feature, Z_f2, z_D2 = model_nn(query_sample_drugs.float())
        # support_sample_drug1=torch.Tensor([np.hstack((embedFeature[x],smileFeature[x])) for x in support_drug1])
        # support_sample_drug2=torch.Tensor([np.hstack((embedFeature[x],smileFeature[x])) for x in support_drug2])
        # query_sample_drug1=torch.Tensor([np.hstack((embedFeature[x],smileFeature[x])) for x in query_drug1])
        # query_sample_drug2=torch.Tensor([np.hstack((embedFeature[x],smileFeature[x])) for x in query_drug2])
        # support_sample_drug1=support_sample_drug1.to(device)
        # support_sample_drug2=support_sample_drug2.to(device)
        # query_sample_drug1=query_sample_drug1.to(device)
        # query_sample_drug2=query_sample_drug2.to(device)
        # support_labels=support_labels.to(device)
        # query_labels=query_labels.to(device)
        # original_support_features = torch.cat((support_sample_drug1,support_sample_drug2),1)
        # original_query_features = torch.cat((query_sample_drug1,query_sample_drug2),1)
        # original_support_features = torch.from_numpy(np.hstack((support_sample_drug1[:,:],support_sample_drug2[:,:])))
        # original_query_features = torch.from_numpy(np.hstack((query_sample_drug1[:, :], query_sample_drug2[:, :])))
        #support_features = feature_encoder(Variable(support_sample_drug1[:,0:400].to(device)),Variable(support_sample_drug1[:,400:].to(device)),Variable(support_sample_drug2[:,0:400].to(device)),Variable(support_sample_drug2[:,400:].to(device)))
        #query_features = feature_encoder(Variable(query_sample_drug1[:,0:400].to(device)),Variable(query_sample_drug1[:,400:].to(device)),Variable(query_sample_drug2[:,0:400].to(device)),Variable(query_sample_drug2[:,400:].to(device)))

        #support_features_ext = support_features.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*NUM_WAYS,1,1)
        #query_features_ext = query_features.unsqueeze(0).repeat(Support_NUM_PER_CLASS*NUM_WAYS,1,1)
        #query_features_ext = torch.transpose(query_features_ext,0,1)
        support_features_ext = mag_support_feature.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS * NUM_WAYS, 1, 1)
        query_features_ext = mag_query_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
        query_features_ext = torch.transpose(query_features_ext, 0, 1)

        #relation_pairs = torch.cat((support_features_ext,query_features_ext),2)
        #print(relation_pairs)
        relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,config['input_dim']*2)
        #print(relation_pairs1)
        relation_pairs = relation_pairs.to(device)
        relations = relation_network(relation_pairs).view(-1,NUM_WAYS)

        loss_r = 0.1 * (F.binary_cross_entropy(recon1, support_sample_drugs.float())+F.binary_cross_entropy(recon2, query_sample_drugs.float()))
        mse = nn.MSELoss().to(device)


        loss_p = 0.1* ( \
                torch.norm(z_D1 - torch.matmul(support_feature, Z_f1)) + \
                0.01 * torch.sum(torch.abs(support_feature)) / (Support_NUM_PER_CLASS * NUM_WAYS) + \
                0.1 * torch.norm(Z_f1, p='fro') / (Support_NUM_PER_CLASS * NUM_WAYS) + \
                torch.norm(z_D2 - torch.matmul(query_feature, Z_f2)) + \
                0.01 * torch.sum(torch.abs(query_feature)) / (QUERY_NUM_PER_CLASS * NUM_WAYS) + \
                0.1 * torch.norm(Z_f2, p='fro') / (QUERY_NUM_PER_CLASS * NUM_WAYS))

        # a1= torch.norm(z_D1 - torch.matmul(support_feature, Z_f1))
        # a2 = 0.01 * torch.sum(torch.abs(support_feature)) / (Support_NUM_PER_CLASS * NUM_WAYS)
        # a3 = 0.1 * torch.norm(Z_f1, p='fro') / (Support_NUM_PER_CLASS * NUM_WAYS)
        # a4 = torch.norm(z_D2 - torch.matmul(query_feature, Z_f2))
        # a5 = 0.01 * torch.sum(torch.abs(query_feature)) / (QUERY_NUM_PER_CLASS * NUM_WAYS)
        # a6 = 0.1 * torch.norm(Z_f2, p='fro') / (QUERY_NUM_PER_CLASS * NUM_WAYS)

        query_labels_array=np.array(query_labels.view(QUERY_NUM_PER_CLASS*NUM_WAYS))
        query_labels_array = (np.arange(query_labels_array.max() + 1) == query_labels_array[:, None]).astype(dtype='float32')
        one_hot_labels = Variable(torch.from_numpy(query_labels_array).to(device))
        query_labels=query_labels.to(device)

        _, predict_labels = torch.max(relations.data, 1)
        train_rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(NUM_WAYS*QUERY_NUM_PER_CLASS)]
        total_train_rewards += np.sum(train_rewards)

        loss_c = mse(relations, one_hot_labels).to(device)


        loss = loss_c + loss_r + loss_p
        #loss = mse(relations, one_hot_labels).to(device)

        #feature_encoder.zero_grad()
        model_nn.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(model_nn.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)


        model_nn_optim.step()
        relation_network_optim.step()
        if (episode + 1) % 500 == 0:
            end = time.time()
            print("episode:", episode + 1, "loss", loss.data)
            train_accuracy = total_train_rewards / 500.0 / NUM_WAYS / QUERY_NUM_PER_CLASS
            print("train_acc:",train_accuracy)
            print("use:" + str(end - start))
            start = time.time()
            if train_accuracy > seen_acc:
                torch.save(model_nn.state_dict(), str(
                    "models1005/model_nn_" + str(NUM_WAYS) + "ways_" + str(Support_NUM_PER_CLASS) + "shot" + str(
                        excludeLabel) + "mag" + str(config['magnify_factor']) + str(config['input_dim']) + "seen.pkl"))
                torch.save(relation_network.state_dict(), str(
                    "models1005/relation_network_" + str(NUM_WAYS) + "ways_" + str(
                        Support_NUM_PER_CLASS) + "shot" + str(excludeLabel) + "mag" + str(
                        config['magnify_factor']) + str(config['input_dim']) + "seen.pkl"))
                seen_acc = train_accuracy

            total_train_rewards=0

        if (episode + 1) % 5000 == 0: #1000
            # test
            print("Testing...")
            total_rewards = 0
            for i in range(TEST_EPISODE): #TEST_EPISODE
                task = tg.MetaDDIETask(CLASS_NUM, NUM_WAYS, Support_NUM_PER_CLASS, Support_NUM_PER_CLASS,"test",excludeLabel)
                #task = tg.MetaDDIETask(CLASS_NUM, NUM_WAYS, Support_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, "test")
                support_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="train",
                                                        shuffle=False)
                query_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="test",
                                                      shuffle=True)
                #query_dataloader = tg.get_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)
                support_drug1, support_drug2, support_labels = support_dataloader.__iter__().next()
                query_drug1, query_drug2, query_labels = query_dataloader.__iter__().next()
                support_sample_drugs = np.zeros((0, config['input_dim']))
                query_sample_drugs = np.zeros((0, config['input_dim']))
                for i in range(Support_NUM_PER_CLASS * NUM_WAYS):
                    support_sample_drugs = np.vstack((support_sample_drugs, np.reshape(
                        smiles2vector(smile[support_drug1[i]], smile[support_drug2[i]]), (1, -1))))
                for i in range(Support_NUM_PER_CLASS * NUM_WAYS):
                    query_sample_drugs = np.vstack((query_sample_drugs, np.reshape(
                        smiles2vector(smile[query_drug1[i]], smile[query_drug2[i]]), (1, -1))))
                support_sample_drugs = torch.from_numpy(support_sample_drugs).to(device)
                query_sample_drugs = torch.from_numpy(query_sample_drugs).to(device)
                recon1, support_feature, mag_support_feature, Z_f1, z_D1 = model_nn(support_sample_drugs.float())
                recon2, query_feature, mag_query_feature, Z_f2, z_D2 = model_nn(query_sample_drugs.float())
                support_features_ext = mag_support_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
                query_features_ext = mag_query_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
                query_features_ext = torch.transpose(query_features_ext, 0, 1)

                support_labels = support_labels.to(device)
                query_labels = query_labels.to(device)


                relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, config['input_dim']*2)

                relation_pairs = relation_pairs.to(device)
                relations = relation_network(relation_pairs).view(-1, NUM_WAYS)
                _, predict_labels = torch.max(relations.data, 1)
                rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(NUM_WAYS)]

                total_rewards += np.sum(rewards)
                #print(sum(rewards))

            test_accuracy = total_rewards / 1.0 / NUM_WAYS / TEST_EPISODE

            print("test accuracy:", test_accuracy)
            print(relations)

            if test_accuracy>last_accuracy:

                torch.save(model_nn.state_dict(),str("models1005/model_nn_"+str(NUM_WAYS)+"ways_"+str(Support_NUM_PER_CLASS)+"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"))
                torch.save(relation_network.state_dict(),str("models1005/relation_network_" + str(NUM_WAYS) + "ways_" + str(Support_NUM_PER_CLASS) + "shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"))

                print("save networks for episode:",episode)

                last_accuracy=test_accuracy

if __name__ == '__main__':
    main()



