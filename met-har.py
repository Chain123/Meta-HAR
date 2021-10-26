import argparse
import copy
import os
import random
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR

import har_model
import utils

cross_entropy = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cosine_similarity = nn.CosineSimilarity()


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2


class reptile_meta(object):

    def __init__(self, graph, lr, device_i, loss_fun, embed_len, number_class, class_map=None, beta=0.5):
        """graph can be norm_embed"""
        super(reptile_meta, self).__init__()
        self.lr = lr
        self.beta = beta
        self.class_map = class_map
        self.number_class = number_class
        # initialize embed model and last layer
        self.model = graph(bidirectional=False)  
        self.model = self.model.to(device_i)
        self.last_layer = har_model.last_layer(embed_len, number_class)
        self.last_layer = self.last_layer.to(device_i)
        self.merge_model = MyEnsemble(self.model, self.last_layer)
        self.training_op = {"last_optimizer": optim.Adam(self.last_layer.parameters(), lr=self.lr, weight_decay=1e-4),
                            "embed_optimizer": optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4),
                            "loss_fun": loss_fun, "loss_fun_last": cross_entropy}
        self.training_res = {"train_acc": [], "test_acc": []}

    def label_transfer(self, target): 
        int_label = target.max(1)[1].numpy()     # [B, 1]
        result = self.class_map[int_label]
        return torch.from_numpy(result)

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
        self.training_op["trainloader"] = utils.dataloader_gen2(self.training_op["train_file"],
                                                               utils.parameter["BATCH_SIZE"] * 2,
                                                               target="hot",
                                                               train=True)
        self.training_op["adaptloader"] = utils.dataloader_gen2(self.training_op["adapt_file"],
                                                               utils.parameter["BATCH_SIZE"] * 2,
                                                               target="hot",
                                                               train=True)
        self.training_op["testloader"] = utils.dataloader_gen2(self.training_op["test_file"], 1, target="hot",
                                                              train=False)
        self.training_op["scheduler"] = StepLR(self.training_op["embed_optimizer"], step_size=2, gamma=0.85)

    def set_train_test_file(self, train, test, adapt):
        self.training_op["train_file"] = train
        self.training_op["test_file"] = test
        self.training_op["adapt_file"] = adapt
        self.user_id = self.training_op["train_file"][0].split(os.sep)[-1].split("_")[0] + "_act"

    def get_model_weights(self):
        return self.model.state_dict()

    def assign_new_weights(self, weights_dict):
        self.model.load_state_dict(weights_dict)

    def train(self, num_epoch):
        # print("=== train on: %s" % self.training_op["train_file"])
        for epoch in range(2 * num_epoch):
            self.model.train()
            train_loss = 0
            for batch_idx, (inputs, targets, _) in enumerate(self.training_op["trainloader"]):
                inputs, targets = inputs.to(device), targets.to(device)
                self.training_op["embed_optimizer"].zero_grad()            
                if device == "cuda":
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_fun"](embeddings, targets, utils.parameter["BATCH_SIZE"])
                else:
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_fun"](embeddings, targets, utils.parameter["BATCH_SIZE"])
                # back propogation
                loss.backward()
                self.training_op["embed_optimizer"].step()
                # loss and accuracy
                train_loss += loss.item()
            self.training_op["scheduler"].step()

    def test_2(self, print_ind=False):
        self.merge_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, targets_t) in enumerate(self.training_op["testloader"]):
            inputs = inputs.to(device)
            targets_t = targets_t.to(device)
            # outputs = self.merge_model(inputs.unsqueeze(1).double())
            if device == "cuda":
                outputs = self.merge_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
            else:
                outputs = self.merge_model(inputs.unsqueeze(1).type(torch.FloatTensor))
            # outputs = self.last_layer(embeddings)
            # _, target_cce = targets.max(1)
            target_cce = torch.from_numpy(all_trans_dict[self.user_id][targets.max(1)[1].numpy()]).to(device)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_t).sum().item()
        if print_ind:
            print("============Test loss: %.3f, Test acc: %.3f" % (
                test_loss / (total + 1), 100.0 * correct / total))
        # self.training_res["test_acc"].append(100.0 * correct / total)
        return 100.0 * correct / total, total

    def test(self, print_ind=False):
        # print("===test on: %s" % self.training_op["test_file"])
        self.model.eval()
        self.last_layer.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, targets_t) in enumerate(self.training_op["testloader"]):
            inputs = inputs.to(device)
            targets_t = targets_t.to(device)
            if device == "cuda":
                embeddings = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
            else:
                embeddings = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))

            outputs = self.last_layer(embeddings)
            # _, target_cce = targets.max(1)
            _, predicted = outputs.max(1)
            target_cce = torch.from_numpy(all_trans_dict[self.user_id][targets.max(1)[1].numpy()]).to(device)
            total += targets.size(0)
            correct += predicted.eq(targets_t).sum().item()
        if print_ind:
            print("============Test loss: %.3f, Test acc: %.3f" % (
                test_loss / (total + 1), 100.0 * correct / total))
        # self.training_res["test_acc"].append(100.0 * correct / total)
        return 100.0 * correct / total, total

    def adapt(self, num_batch, train=False):
        """
        First Adapt embedding net, then fine-tune the last layer.
        """
        # Fine-tune the embedding net
        self.model.train()
        if train:
            adapt_loader = self.training_op["trainloader"]
        else:
            adapt_loader = self.training_op["adaptloader"]

        for batch_idx, (inputs, targets, targets_t) in enumerate(adapt_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.training_op["embed_optimizer"].zero_grad()
            if device == "cuda":
                embeddings = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                loss = self.training_op["loss_fun"](embeddings, targets, int(utils.parameter["BATCH_SIZE"] / 2))
            else:
                embeddings = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                loss = self.training_op["loss_fun"](embeddings, targets, int(utils.parameter["BATCH_SIZE"] / 2))

            loss.backward()
            self.training_op["embed_optimizer"].step()

        # Fine-tune last layer
        self.last_layer.train()
        correct = 0
        total = 0
        last_opt = optim.Adam(self.last_layer.parameters(), lr=self.lr, weight_decay=1e-4)
        for epoch in range(num_batch):
            for batch_idx, (inputs, targets, targets_t) in enumerate(adapt_loader):
                inputs, targets, targets_t = inputs.to(device), targets.to(device), targets_t.to(device)
                # self.training_op["last_optimizer"].zero_grad()
                last_opt.zero_grad()
                if device == "cuda":
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                else:
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                embeddings.detach()   # TODO detach or not?
                outputs = self.last_layer(embeddings)
                if device == "cuda":
                    loss = self.training_op["loss_fun_last"](outputs, targets_t.type(torch.cuda.LongTensor) )
                else:
                    loss = self.training_op["loss_fun_last"](outputs, targets_t.type(torch.LongTensor))
                loss.backward()
                last_opt.step()

    def adapt_fixed(self, num_batch, train=False):
        """
        First Adapt embedding net, then fine-tune the last layer.
        """
        # Fine-tune the embedding net
        self.model.train()
        if train:
            adapt_loader = self.training_op["trainloader"]
        else:
            adapt_loader = self.training_op["adaptloader"]
        # Fine-tune last layer
        self.last_layer.train()
        correct = 0
        total = 0
        last_opt = optim.Adam(self.last_layer.parameters(), lr=self.lr, weight_decay=1e-4)
        for epoch in range(num_batch):
            for batch_idx, (inputs, targets) in enumerate(adapt_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # self.training_op["last_optimizer"].zero_grad()
                last_opt.zero_grad()
                embeddings = self.model(inputs.unsqueeze(1).double())
                embeddings.detach()
                # print(embeddings.size())
                outputs = self.last_layer(embeddings)
                if device == "cuda":
                    loss = self.training_op["loss_fun_last"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
                else:
                    loss = self.training_op["loss_fun_last"](outputs, targets.max(1)[1].type(torch.LongTensor))
                loss.backward()
                last_opt.step()
                # fine_tune acc
                _, predicted = outputs.max(1)
                _, target_cce = targets.max(1)
                total += targets.size(0)
                correct += predicted.eq(target_cce).sum().item()

    def asign_merge_model(self):
        pass

    def adapt_merged(self, num_batch, train=False):
        """
        First Adapt embedding net, then fine-tune the last layer.
        """
        # Fine-tune the embedding net
        self.merge_model.train()
        if train:
            adapt_loader = self.training_op["trainloader"]
        else:
            adapt_loader = self.training_op["adaptloader"]
        # Fine-tune last layer
        correct = 0
        total = 0
        last_opt = optim.Adam(self.merge_model.parameters(), lr=self.lr, weight_decay=1e-4)
        for epoch in range(num_batch):
            for batch_idx, (inputs, targets) in enumerate(adapt_loader):
                targets = torch.from_numpy(all_trans_dict[self.user_id][targets.max(1)[1].numpy()])
                inputs, targets = inputs.to(device), targets.to(device)
                # self.training_op["last_optimizer"].zero_grad()
                last_opt.zero_grad()
                outputs = self.merge_model(inputs.unsqueeze(1).double())
                # print(embeddings.size())
                # outputs = self.last_layer(embeddings)
                if device == "cuda":
                    loss = self.training_op["loss_fun_last"](outputs, targets.type(torch.cuda.LongTensor))
                    # loss = self.training_op["loss_fun_last"](outputs, targets.max(1)[1].type(torch.cuda.LongTensor))
                else:
                    loss = self.training_op["loss_fun_last"](outputs, targets.type(torch.LongTensor))
                loss.backward()
                last_opt.step()

    def adapt_merged_2(self, num_batch, train=False):
        """
        First Adapt embedding net, then fine-tune the last layer.
        """
        # Fine-tune the embedding net
        self.merge_model.train()
        if train:
            adapt_loader = self.training_op["trainloader"]
        else:
            adapt_loader = self.training_op["adaptloader"]
        # Fine-tune last layer
        correct = 0
        total = 0
        last_opt = optim.Adam(self.merge_model.parameters(), lr=utils.parameter["lr"], weight_decay=1e-4)
        self.model.train()
        for embed_epoch in range(10 * num_batch):
            for batch_idx, (inputs, targets, _) in enumerate(adapt_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                self.training_op["embed_optimizer"].zero_grad()        
                if device == "cuda":
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_fun"](embeddings, targets, int(utils.parameter["BATCH_SIZE"] / 2))
                else:
                    embeddings = self.model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_fun"](embeddings, targets, int(utils.parameter["BATCH_SIZE"] / 2))
                
                loss.backward()
                self.training_op["embed_optimizer"].step()

        for epoch in range(num_batch * 10):
            for batch_idx, (inputs, targets, targets_t) in enumerate(adapt_loader):
                # targets = torch.from_numpy(all_trans_dict[self.user_id][targets.max(1)[1].numpy()])
                inputs, targets, targets_t = inputs.to(device), targets.to(device), targets_t.to(device)
                # self.training_op["last_optimizer"].zero_grad()
                last_opt.zero_grad()
                
                if device == "cuda":
                    outputs = self.merge_model(inputs.unsqueeze(1).type(torch.cuda.FloatTensor))
                    loss = self.training_op["loss_fun_last"](outputs, targets_t.type(torch.cuda.LongTensor))
                    # targets.max(1)[1].type(torch.cuda.LongTensor)
                else:
                    outputs = self.merge_model(inputs.unsqueeze(1).type(torch.FloatTensor))
                    loss = self.training_op["loss_fun_last"](outputs, targets_t.type(torch.LongTensor))
                loss.backward()
                last_opt.step()
                

def update_server_weights(w_list, w, sigma=0.2):
    """
    model 1 and model 2 with same structure
    return weights dict with values w_model1 - w_model2

    sigma = 1 : federated learning
    """
    w_avg = copy.deepcopy(w_list[0])
    for k in w.keys():
        for idx in range(1, len(w_list)):
            w_avg[k] += w_list[idx][k]
        w_avg[k] = w[k] + torch.mul((torch.div(w_avg[k], len(w_list)).sub(w[k])), sigma)
    return w_avg


def update_server_weights_weighted(w_list, w, sigma=0.2):  # TO BE DONE
    # weights:
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


def main(rounds, data_dir, out_dir, lr=0.001, local_e=1, leave_out=None, sigma=0.1, all_trans_dict=None):
    """
    :param rounds: global rounds for federated learning: type: float
    :param out_dir: output result dir: type: string
    :param lr: initial learning rate: type float
    :param local_e: local update epochs for federated learning: type: int
    :param leave_out: leave out user index: 0-8 type: int
    :param sigma: w_new = w + sigma* (avg_updated_w - w): sigma: float 0-1
    :return: None
    Save test acc result in output dir:
        1. model acc on leave out user: before and after adapt
        2. model acc on fed users: before and after adapt
    """
    if leave_out is None:
        leave_out = [0]

    # all_act_num = utils.load_pickle("/data/ceph/seqrec/fl_data/www21/source/num_act_user_all.pickle")
    all_act_num = {}
    for user in all_trans_dict.keys():
        all_act_num[user] = np.sum(all_trans_dict[user] != -1)  # number of local activities of `user'
    # the number of local act for each users.

    all_train = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if "train" in file]
    all_test = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'test' in file]
    random.Random(0).shuffle(all_train)
    random.Random(0).shuffle(all_test)

    # 5 leave out for meta testing
    train_users_train = all_train[0:-5]
    train_user_test = all_test[0:-5]
    test_users_train = all_train[-5:]
    test_user_test = all_test[-5:]

    # Meta-train users
    client_models = []
    for ind in range(len(train_users_train)):
        file_name = train_users_train[ind].split(os.sep)[-1].split("_")[0]
        client_models.append(
            reptile_meta(har_model.norm_embed, lr, device, utils.pairwiseloss(), 100, all_act_num[file_name]))
        client_train_file = [train_users_train[ind]]
        client_test_file = [train_user_test[ind]]
        client_adapt_file = client_train_file

        client_models[-1].set_train_test_file(client_train_file, client_test_file, client_adapt_file)
        client_models[-1].build_data_loader()

    # Meta-test users
    # Note: By replacing `norm_embed' with the `norm_cce' and removing the local fine-tune we get the reptile method.
    leave_out_modules = []
    for ind in range(5):
        file_name = test_users_train[ind].split(os.sep)[-1].split("_")[0]
        leave_out_modules.append(
            reptile_meta(har_model.norm_embed, lr, device, utils.pairwiseloss(), 100, all_act_num[file_name]))
        leave_train = [test_users_train[ind]]
        leave_test = [test_user_test[ind]]
        leave_adapt_file = leave_train
        leave_out_modules[-1].set_train_test_file(leave_train, leave_test, leave_adapt_file)
        leave_out_modules[-1].build_data_loader()

    # server model
    server_model = reptile_meta(har_model.norm_embed, lr, device, utils.pairwiseloss(), 100, 7)
    # there are totally 7 activities in global act set.
    # server_model result dir
    fed_test_result = {"before": []}    # result before fine-tune
    leave_test_result = {"before": []}
    init_after = 0
    for val in [1, 1, 1]:    # result after x steps of local fine-tune
        fed_test_result["tune_%d" % (init_after + val)] = []
        leave_test_result["tune_%d" % (init_after + val)] = []
        init_after += val

    for i in range(rounds):  # global rounds
        # sampling tasks/ here is the users
        chosen_client = np.random.choice(len(train_users_train), 5, replace=False)
        updated_weights = []
        for user_idx in chosen_client:
            # pull weights theta from center
            ###############################
            # print(server_model.get_model_weights().keys())
            # sys.exit()
            ###############################
            print("      -- Start local training on user:  ", train_users_train[user_idx])
            client_models[user_idx].assign_new_weights(server_model.get_model_weights())
            # local train
            client_models[user_idx].train(num_epoch=local_e)
            # get updated para difference.
            updated_weights.append(client_models[user_idx].get_model_weights())
            # print("done local train on user: %s" % train_users[user_idx])

        print("# update global meta model ========")
        server_model.assign_new_weights(
            update_server_weights(updated_weights, server_model.get_model_weights(), sigma=sigma))

        if i > 70 and i % 2 == 0:
            print("# ====== Testing ===========")
            adapt_val = 0
            for val in [1, 1, 1]:
                adapt_acc = []
                weights = []
                for user_idx in range(len(train_users_train)):
                    client_models[user_idx].assign_new_weights(server_model.get_model_weights())
                    client_models[user_idx].adapt_merged_2(num_batch=val + adapt_val, train=False)
                    acc, num = client_models[user_idx].test()
                    adapt_acc.append(acc)
                    weights.append(num)

                adapt_val += val
                fed_test_result["tune_%d" % adapt_val].append(np.average(adapt_acc, weights=weights))

            # Meta-test users
            print("# Testing meta-test user ===========")
            adapt_val = 0
            for val in [1, 1, 1]:
                adapt_acc = []
                weights = []
                for j in range(5):
                    leave_out_modules[j].assign_new_weights(server_model.get_model_weights())
                    leave_out_modules[j].adapt_merged_2(num_batch=val + adapt_val, train=False)
                    acc, num = leave_out_modules[j].test()
                    adapt_acc.append(acc)
                    weights.append(num)
                adapt_val += val
                leave_test_result["tune_%d" % adapt_val].append(np.average(adapt_acc, weights=weights))
            fed_test_file = os.path.join(out_dir, "metahar_train_b2.pickle")
            leave_test_file = os.path.join(out_dir, "metahar_test_b2.pcikle")
            utils.save_pickle(fed_test_result, fed_test_file)
            utils.save_pickle(leave_test_result, leave_test_file)

    # save test results	local_sigma_tmp_leave
    leave_str = ""
    for val in leave_out:
        leave_str += str(val)
    fed_test_file = os.path.join(out_dir, "metahar_train_b2.pickle")   # meta-train users
    leave_test_file = os.path.join(out_dir, "metahar_test_b2.pcikle")  # meta-test users
    utils.save_pickle(fed_test_result, fed_test_file)
    utils.save_pickle(leave_test_result, leave_test_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-HAR')
    parser.add_argument('--dataset', type=str, help='Dataset dir',
                        default="F:\\www21\\final_version\\Meta-HAR\\Data\\collected_pickle")
    parser.add_argument('--result_dir', type=str, help='result dir',
                        default="F:\\www21\\final_version\\Meta-HAR\\results")
    parser.add_argument('--local_e', type=int, default=2,
                        help='number of local epochs in the federated training phase')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma (refer to the paper)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--adapt_num', type=int, default=5,
                        help='number of adaptation steps (in the fine-tune phase)')
    args = parser.parse_args()

    all_trans_dict = utils.load_pickle("F:\\www21\\data\\trans_dict_collect.pickle")
    main(100, args.dataset, args.result_dir, lr=args.lr, local_e=args.local_e, sigma=args.sigma,
         all_trans_dict=all_trans_dict)
    # total update rounds=100
