'''
Train central classification model. 
filelists of training and testing file names.
'''

import numpy as np
import argparse
import torch
import utils
import os
import plot
# import tensorboardX
import har_model
import torch.optim as optim
from torch import nn
from tqdm import tqdm
# import _pcikle as pickle
# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Training embedding net on sensor data')
parser.add_argument('--data', help='Dataset dir')
parser.add_argument('--out', help='output dir')
parser.add_argument('--epoches', type=int, help='output dir')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
out_path = args.out

# writer = SummaryWriter(logdir=out_path)


class central_model(object):

    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = har_model.norm_cce(bidirectional=False, num_classes=7) 
        self.model = self.model.to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=utils.parameter["lr"], weight_decay=1e-4)
        self.cross_entropy = nn.CrossEntropyLoss()

    def valid(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.opt.zero_grad()        
            if device == "cuda":
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.cross_entropy(outputs, targets.type(torch.cuda.LongTensor))  # .max(1)[1].type()
            else:
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.cross_entropy(outputs, targets.type(torch.LongTensor))        
            # Current training preformance
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size()[0]  # （batch size）
            correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        mean_loss = epoch_loss / total
        print("# test acc", acc, "test loss", mean_loss)
        return acc

    def train(self, epoches):
        self.model.train()
        for round_num in range(epoches):
            correct = 0
            total = 0
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                self.opt.zero_grad()
                # print(inputs.type)
                if device == "cuda":
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))  
                    loss = self.cross_entropy(outputs, targets.type(torch.cuda.LongTensor))  # .max(1)[1].type()
                else:
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))  
                    loss = self.cross_entropy(outputs, targets.type(torch.LongTensor))
                # back-propogation
                loss.backward()
                self.opt.step()

                # Current training preformance
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                # print(targets.size())
                total += targets.size()[0]  # （batch size）
                correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            mean_loss = epoch_loss / total
            if round_num % 5 == 0:
                print("training epoch", round_num)
                print("train acc", acc, "train loss", mean_loss)
        return self.valid(round_num)

    
def save_pickle(data_dict, filename):
    with open(filename, "wb") as myfile:
        pickle.dump(data_dict, myfile, -1)    
       
     
def load_pickle(filename, show_name=False):
    if show_name:
        print(filename)
    return pickle.load(open(filename, "rb"))

    
def self_test_all(in_dir):
    filename = ["_".join(file.split("_")[0:2]) for file in os.listdir(in_dir) if "train" in file]
    result = {}
    for file in filename:
        print("user:", file)
        train_file = os.path.join(in_dir, file + "_train.pickle")
        test_file = os.path.join(in_dir, file + "_test.pickle")
        print(train_file)
        trainloader = utils.dataloader_gen([train_file], utils.parameter["BATCH_SIZE"], target="number")
        testloader = utils.dataloader_gen([test_file], 1, train=False, target="number")
        model = central_model(trainloader, testloader)
        test_acc = model.train(args.epoches)
        result[file] = test_acc
    print(result)
    with open("self_test.txt", "w") as myfile:        
        for key in result:
            line = key + "\t" + str(result[key]) + "\n"
            myfile.write(line)
    # save_pickle(result, "self_test.pickle")
    

def all_central():
    # data loader
    collect_dir = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/collect"
    my_dir = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/mine"
    collect_users = load_pickle("final_selected_user_collect.pickle")
    my_users = load_pickle("final_selected_user_mine.pickle")
    collected_train = [os.path.join(collect_dir, file + "_train.pickle") for file in collect_users]
    collected_test = [os.path.join(collect_dir, file + "_test.pickle") for file in collect_users]
    my_train = [os.path.join(my_dir, file + "_train.pickle") for file in my_users]
    my_test = [os.path.join(my_dir, file + "_test.pickle") for file in my_users]
    
    trainloader = utils.dataloader_gen(collected_train + my_train, utils.parameter["BATCH_SIZE"], target="number")
    testloader = utils.dataloader_gen(collected_test + my_test, 1, train=False, target="number")
    # model and train
    model = central_model(trainloader, testloader)
    model.train(args.epoches)
    
# main()
# self_test_all(args.data)
all_central()