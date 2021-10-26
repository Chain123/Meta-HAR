import _pickle as pickle
import argparse
import copy
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR

import har_model
import utils

# from torch.nn import PairwiseDistance


cross_entropy = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cosine_similarity = nn.CosineSimilarity()

parser = argparse.ArgumentParser(description='Training embedding net on sensor data')
parser.add_argument('--dataset', help='Dataset dir')
parser.add_argument('--result_dir', help='result dir')
parser.add_argument('--local_e', type=int, help='federated local epoches')
parser.add_argument('--sigma', type=float, help='sigma')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--adapt_num', default=5, type=int, help='learning rate')
args = parser.parse_args()


def load_pickle(filename, show_name=False):
    if show_name:
        print(filename)
    return pickle.load(open(filename, "rb"))


def save_pickle(data_dict, filename):
    with open(filename, "wb") as myfile:
        pickle.dump(data_dict, myfile, -1)


class reptile_meta(object):

    def __init__(self, graph, lr, device, loss_fun, number_class, beta=0.5):
        """graph can be norm_cce, merge_cce"""
        super(reptile_meta, self).__init__()
        self.lr = lr
        self.beta = beta
        self.model = graph(bidirectional=False, num_classes=number_class)  # cross entropy based model
        self.model = self.model.to(device)
        # self.model = self.model.double()
        self.training_op = {}
        self.training_op["optimizer"] = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # self.training_op["optimizer"] = optim.SGD(self.model.parameters(),
        # lr= self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.training_op["scheduler"] = StepLR(self.training_op["optimizer"], step_size=2, gamma=0.85)
        self.training_op["loss_fun"] = loss_fun
        self.training_res = {"train_acc": [], "test_acc": []}

    def save_model(self, filename):
        """ filename example: /path/checkpoint/model-100.t7
            State dict contains: "model" key at least.
        """
        state = {
            "model": self.model.state_dict(),
            "client": self.model_name,
        }
        torch.save(state, filename)

    def build_data_loader(self):
        self.training_op["trainloader"] = utils.dataloader_gen(self.training_op["train_file"],
                                                               utils.parameter["BATCH_SIZE"],
                                                               target="hot")      # (one_hot?)
        self.training_op["adaptloader"] = utils.dataloader_gen(self.training_op["adapt_file"],
                                                               utils.parameter["BATCH_SIZE"],
                                                               target="hot")
        self.training_op["testloader"] = utils.dataloader_gen(self.training_op["test_file"], 1, target="hot")

    def set_train_test_file(self, train, test, adapt):
        self.training_op["train_file"] = train
        self.training_op["test_file"] = test
        self.training_op["adapt_file"] = adapt

    def get_model_weights(self):
        return self.model.state_dict()

    def assign_new_weights(self, weights_dict):
        self.model.load_state_dict(weights_dict)

    def train(self, num_epoch, global_center=None):
        # print("=== train on: %s" % self.training_op["train_file"])
        for epoch in range(num_epoch):
            self.model.train()
            correct = 0
            total = 0
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.training_op["trainloader"]):
                inputs, targets = inputs.to(device), targets.to(device)
                self.training_op["optimizer"].zero_grad()
                # add the channel dimension, return logits and embedding for cce model
                # outputs, _ = self.model(inputs.unsqueeze(1).double())
                if device == "cuda":
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
                    # as here we use targets.max() the target should be in one-hot form.
                else:
                    outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.LongTensor))
                # back propogation
                loss.backward()
                self.training_op["optimizer"].step()
                # loss and accuracy
                train_loss += loss.item()
                _, target_cce = targets.max(1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(target_cce).sum().item()
            print("===== local epoch: %d =======" % epoch)
            print("train loss % .3f, train acc: % .3f ,lr: %f " % (
                train_loss / (batch_idx + 1), 100.0 * correct / total, self.training_op["scheduler"].get_lr()[0]))

            # learning rate
            self.training_op["scheduler"].step()
        # self.test()

    def test(self, print_ind=False):
        # print("===test on: %s" % self.training_op["test_file"])
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.training_op["testloader"]):
            inputs, targets = inputs.to(device), targets.to(device)
            if device == "cuda":
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
            else:
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.LongTensor))
            test_loss += loss.item()
            _, target_cce = targets.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_cce).sum().item()
        if print_ind:
            print("============Test loss: %.3f, Test acc: %.3f" % (
                test_loss / (total + 1), 100.0 * correct / total))
        return 100.0 * correct / total, total

    def adapt(self, num_batch):
        """
        Used for test only
        """
        self.model.train()
        correct = 0
        total = 0
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.training_op["adaptloader"]):
            inputs, targets = inputs.to(device), targets.to(device)
            self.training_op["optimizer"].zero_grad()
            # torch cross entropy
            if device == "cuda":
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
            else:
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.LongTensor))
            # back propogation
            loss.backward()
            self.training_op["optimizer"].step()
            # loss and accuracy
            train_loss += loss.item()
            _, target_cce = targets.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_cce).sum().item()

    def adapt_train(self):
        self.model.train()
        correct = 0
        total = 0
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.training_op["trainloader"]):
            inputs, targets = inputs.to(device), targets.to(device)
            self.training_op["optimizer"].zero_grad()        
            if device == "cuda":
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
            else:
                outputs, _ = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.training_op["loss_fun"](outputs, targets.max(1)[1].type(torch.LongTensor))
            
            loss.backward()
            self.training_op["optimizer"].step()
            
            # loss and accuracy
            train_loss += loss.item()
            _, target_cce = targets.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_cce).sum().item()


def update_server_weights(w_list, w, sigma=0.2):
    """
    model_1 and model_2 with same structure
    return weights dictionary with values w_model_1 - w_model_2

    sigma = 1 : federated learning
    """
    w_avg = copy.deepcopy(w_list[0])
    for k in w.keys():
        for idx in range(1, len(w_list)):
            w_avg[k] += w_list[idx][k]
        w_avg[k] = w[k] + torch.mul((torch.div(w_avg[k], len(w_list)).sub(w[k])), sigma)
    return w_avg


def update_server_weights_weighted(w_list, w, sigma=0.2): 
    """
    abandon: more complicated way to update the model on the server side.   
    """
    # flatten all model weights
    weights_flattened = []
    keys = w.keys()
    used_keys = []
    for ind, key in enumerate(keys):
        if "batches_tracked" not in key:
            used_keys.append(key)
    # flatten w
    w_param = []
    for key in used_keys:
        w_param.append(torch.flatten(w[key]))
    w_flatten = torch.cat(w_param, dim=0)
    # print(w_flatten.size())
    # flatten w_list
    for idx in range(len(w_list)):
        model_params = []
        for key in used_keys:
            model_params.append(torch.flatten(w_list[idx][key]))
        weights_flattened.append(torch.cat(model_params, dim=0))
    # calculate l2 norm
    l2 = []
    cosine_dis = []
    for idx in range(len(w_list)):
        if device == "cpu":
            l2.append(np.squeeze(torch.dist(weights_flattened[idx], w_flatten).numpy()))
            cosine_dis.append(np.squeeze(
                cosine_similarity(weights_flattened[idx].unsqueeze(dim=0), w_flatten.unsqueeze(dim=0)).numpy()))
        else:
            l2.append(np.squeeze(torch.dist(weights_flattened[idx], w_flatten).cpu().numpy()))
            cosine_dis.append(np.squeeze(
                cosine_similarity(weights_flattened[idx].unsqueeze(dim=0), w_flatten.unsqueeze(dim=0)).cpu().numpy()))
    weights = []
    for index in range(len(w_list)):
        # weights.append(np.sqrt(l2[index] ** 2 + cosine_dis[index] ** 2))
        weights.append(l2[index] * np.abs(cosine_dis[index]))
    # normalize weights
    weights_norm = []
    total = np.sum(weights)
    for val in weights:
        weights_norm.append(val / total)
    ###################
    # print(weights_norm)
    # sys.exit()
    ###################
    # weighted average
    w_avg = copy.deepcopy(w_list[0])
    for k in w.keys():
        w_avg[k] = w_avg[k] * weights_norm[0]
    for k in w.keys():
        for idx in range(1, len(w_list)):
            w_avg[k] += w_list[idx][k] * weights_norm[idx]
        w_avg[k] = w[k] + torch.mul(w_avg[k].sub(w[k]), sigma)

    return w_avg


def main(rounds, out_dir, lr=0.001, local_e=1, leave_out=[0], sigma=0.1, tmp=0):
    """
    :param rounds: global rounds for federated learning: type: float
    :param in_dir: input data dir: type: string
    :param out_dir: output result dir: type: string
    :param lr: initial learning rate: type float
    :param local_e: local update epoches for federated learning: type: int
    :param adapt_num: number of batch go through when used for adaptation: type: int
    :param leave_out: leave out user index: 0-8 type: int
    :param sigma: w_new = w + sigma* (avg_updated_w - w): sigma: float 0-1
    :return: None
    Save test acc result in output dir:
        1. model acc on leave out user: before and after adapt
        2. model acc on fed users: before and after adapt
    """

    train_users = load_pickle("../final_selected_user_collect.pickle")  # meta-train users
    leave_users = load_pickle("../final_selected_user_mine.pickle")     # meta-test users
    collect_dir = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/collect"  # data dir part1
    my_dir = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/mine"          # data dir part2

    # Meta-train set-up
    print("# meta-train user setup ...", len(train_users))
    client_models = []
    for val in train_users:
        client_models.append(reptile_meta(har_model.norm_cce, lr, device, cross_entropy, number_class=7))
        
        # with the cross_entropy loss, this is the reptile model, when there is no fine-tune the results 
        # is the same as the fedavg. 

        client_train_file = [os.path.join(collect_dir, val + "_train.pickle")]
        client_test_file = [os.path.join(collect_dir, val + "_test.pickle")]
        # client_adapt_file = os.path.join(in_dir, val, "stress/fine_tune_data")
        client_adapt_file = client_train_file
        client_models[-1].set_train_test_file(client_train_file, client_test_file, client_adapt_file)
        client_models[-1].build_data_loader()

    # build result dir
    # Meta-test users
    print("# meta-train user setup ...", len(leave_users))
    leave_out_modules = []
    for leave_user_name in leave_users:
        leave_out_modules.append(reptile_meta(har_model.norm_cce, lr, device, cross_entropy, number_class=7))
        leave_train = [os.path.join(my_dir, leave_user_name + "_train.pickle")]  # fine-tune data
        leave_test = [os.path.join(my_dir, leave_user_name + "_test.pickle")]    # test data
        # leave_adapt_file = os.path.join(in_dir, leave_user_name, "stress/fine_tune_data")
        leave_adapt_file = leave_train
        leave_out_modules[-1].set_train_test_file(leave_train, leave_test, leave_adapt_file)
        leave_out_modules[-1].build_data_loader()

    # server model for meta updating
    server_model = reptile_meta(har_model.norm_cce, lr, device, cross_entropy, number_class=7)
    
    # server_model result dir    
    fed_test_result = {"before": []}        # result  before personalization
    leave_test_result = {"before": []}
    init_after = 0
    for val in [1, 1, 1]:                   # different number of fine-tune steps.
        fed_test_result["tune_%d" % (init_after + val)] = []
        leave_test_result["tune_%d" % (init_after + val)] = []
        # leave_test_result["after_%d_1" % (init_after + val)] = []
        init_after += val

    for i in range(rounds):  # global rounds
        # sampling users 
        chosen_client = np.random.choice(len(train_users), 5, replace=False)
        updated_weights = []
        for user_idx in chosen_client:
            # pull weights theta from center
            client_models[user_idx].assign_new_weights(server_model.get_model_weights())
            # local train
            client_models[user_idx].train(num_epoch=local_e)
            # get updated para difference.
            updated_weights.append(client_models[user_idx].get_model_weights())
            print("done local train on user: %s" % train_users[user_idx])

        print("======= update global meta model ========")
        server_model.assign_new_weights(
            update_server_weights(updated_weights, server_model.get_model_weights(), sigma=sigma))

        if i > 30:  # start to evaluate current model after 30 global rounds.
            print("========== Testing ===========")
            # Meta-train users
            before_test = []
            before_num = []
            for user_idx in range(len(train_users)):
                client_models[user_idx].assign_new_weights(server_model.get_model_weights())
                acc, num = client_models[user_idx].test()
                before_test.append(acc)
                before_num.append(num)
            fed_test_result["before"].append(np.average(before_test, weights=before_num))

            adapt_val = 0
            for val in [1, 1, 1]:   # three adapt steps
                adapt_acc = []
                weights = []
                for user_idx in range(len(train_users)):
                    # client_models[user_idx].assign_new_weights(server_model.get_model_weights())
                    print("======user: %d, adapt: %d" % (user_idx, adapt_val + val))
                    client_models[user_idx].adapt_train()
                    acc, num = client_models[user_idx].test()
                    adapt_acc.append(acc)
                    weights.append(num)
                fed_test_result["tune_%d" % (adapt_val + val)].append(np.average(adapt_acc, weights=weights))
                adapt_val += val

            # Meta-test users
            before_test = []
            before_num = []
            for j in range(len(leave_users)):
                # for every Meta-test user
                leave_out_modules[j].assign_new_weights(server_model.get_model_weights())
                acc, num = leave_out_modules[j].test()
                before_test.append(acc)
                before_num.append(num)
            leave_test_result["before"].append(np.average(before_test, weights=before_num))

            # for different number of adapt batch
            adapt_val = 0
            for val in [1, 1, 1]:
                adapt_acc = []
                weights = []
                for j in range(len(leave_users)):
                    leave_out_modules[j].adapt(num_batch=val)
                    acc, num = client_models[user_idx].test()
                    adapt_acc.append(acc)
                    weights.append(num)
                leave_test_result["tune_%d" % (adapt_val + val)].append(np.average(adapt_acc, weights=weights))
                adapt_val += val

    # save test results	local_sigma_tmp_leave
    leave_str = ""
    for val in leave_out:
        leave_str += str(val)
    fed_test_file = os.path.join(out_dir,
                                 "fed_test_N0_%d_%.2f_%d_%s" % (local_e, sigma, tmp, leave_str))
    leave_test_file = os.path.join(out_dir,
                                   "leave_test_N0_%d_%.2f_%d_%s" % (local_e, sigma, tmp, leave_str))
    utils.save_pickle(fed_test_result, fed_test_file)
    utils.save_pickle(leave_test_result, leave_test_file)


if __name__ == '__main__':
    main(100, "result", lr=utils.parameter["lr"], local_e=2, sigma=1.0)
