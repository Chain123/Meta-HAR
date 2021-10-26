"""
Train central classification model.
An early stop implementation https://github.com/Bjarten/early-stopping-pytorch can also be applied here.
"""

import argparse
import os

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

# import tensorboardX
import har_model
import utils

import _pickle as pickle

# from torch.utils.tensorboard import SummaryWriter


class central_model(object):

    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = har_model.norm_cce(bidirectional=False, num_classes=7)
        self.model = self.model.to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=utils.parameter["lr"], weight_decay=1e-4)
        self.cross_entropy = nn.CrossEntropyLoss()

    def valid(self):
        self.model.eval()
        correct = 0
        total = 0
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.opt.zero_grad()
            if device == "cuda":
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.cross_entropy(outputs, targets.type(torch.cuda.LongTensor))     # .max(1)[1].type()
            else:
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.cross_entropy(outputs, targets.type(torch.LongTensor))
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size()[0]
            correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        print(f" === test acc: {acc}")
        return acc

    def train_step(self, epochs):
        self.model.train()
        for round_num in range(epochs):
            correct = 0
            total = 0
            epoch_loss = 0.0

            pbar = tqdm(enumerate(self.trainloader))
            pbar.set_description(f'[Train epoch {round_num}]')
            # for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            for batch_idx, data_batch in pbar:
                inputs, targets = data_batch[0], data_batch[1]
                inputs, targets = inputs.to(device), targets.to(device)
                self.opt.zero_grad()
                if device == "cuda":
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.cross_entropy(outputs, targets.type(torch.cuda.LongTensor))  # .max(1)[1].type()
                else:
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.cross_entropy(outputs, targets.type(torch.LongTensor))
                # back
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size()[0]  # （batch size）
                correct += predicted.eq(targets).sum().item()
                acc_step = 100.0 * predicted.eq(targets).sum().item() / targets.size()[0]
                pbar.set_postfix(accuracy=acc_step, ce_loss=loss.item())
            # epoch metric
            acc = 100.0 * correct / total
            print(f"epoch {round_num}: epoch acc: {acc}")
            self.valid()
        # return self.valid()


def save_pickle(data_dict, filename):
    with open(filename, "wb") as fid:
        pickle.dump(data_dict, fid, -1)


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
        test_acc = model.train_step(args.epochs)
        result[file] = test_acc
    print(result)
    with open("self_test.txt", "w") as fid:
        for key in result:
            line = key + "\t" + str(result[key]) + "\n"
            fid.write(line)
    # save_pickle(result, "self_test.pickle")


def all_central():
    # data loader
    data_dir = "F:\\www21\\final_version\\Meta-HAR\\Data\\collected_pickle"
    train_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'train' in file]
    test_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'test' in file]
    print(len(train_files), len(test_files))
    trainloader = utils.dataloader_gen(train_files, utils.parameter["BATCH_SIZE"], target="number")
    testloader = utils.dataloader_gen(test_files, 1, train=False, target="number")
    # model and train
    model = central_model(trainloader, testloader)
    model.train_step(args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training embedding net on sensor data')
    parser.add_argument('--data', help='Dataset dir')
    parser.add_argument('--out', help='output dir')
    parser.add_argument('--epochs', type=int, default=10,
                        help='max number of training epoch')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    out_path = args.out

    all_central()
    # self_test_all(args.data)
