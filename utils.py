import _pickle as pickle
import torch
import torch.nn as nn
import sys
import data_loader
import numpy as np
import time

global parameter
parameter = {}
parameter["CONV_LEN"] = 3
parameter["CONV_LEN_INTE"] = 3
parameter["CONV_LEN_LAST"] = 2
parameter["CONV_NUM"] = 64
parameter["CONV_MERGE_LEN"] = 2
parameter["CONV_MERGE_LEN2"] = 6
parameter["CONV_MERGE_LEN3"] = 4
parameter["CONV_NUM2"] = 64
parameter["INTER_DIM"] = 100
parameter["NUM_SENSOR"] = 2
parameter["CONV_KEEP_PROB"] = 0.8

parameter["SEQUENCE_LEN"] = 12
parameter["SAMPLE_INTERVAL_LEN"] = 7
parameter["ACT_OUT_DIM"] = 7

parameter["BATCH_SIZE"] = 128   # too small.
parameter["BATCH_NUM"] = 28000
parameter["EPOCH"] = 100
parameter["NUM_DIM"] = 8
parameter["radius"] = 10
parameter["tao"] = 0.0005  # If only last layer is included for the regularizer
parameter["mu"] = 0.5
parameter["Augment"] = 10
parameter["lr"] = 0.001

softplus = nn.Softplus()
cosine_similarity = nn.CosineSimilarity()
cross_entropy = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dataloader_gen(filename, batch_size, train=True, target="logits"):
    """
        Args:
            filename: a list of pickle files that stores the processed HAR data (after FFT)
            batch_size: batch size
            train: is for training?
            target: encoder method for the label: "hot": one-hot encoding, else int number
    """
    dataset = data_loader.heter_data(filename, target=target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2,
                                             drop_last=True)
    return dataloader


def dataloader_gen2(filename, batch_size, train=True, target="logits"):
    dataset = data_loader.heter_data2(filename, target=target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2,
                                             drop_last=True)
    return dataloader


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f, encoding="latin")


def save_pickle(dict_name, file_name):
    with open(file_name, "wb") as myfile:
        pickle.dump(dict_name, myfile)


class pairwiseloss(torch.nn.Module):
    def __init__(self):
        super(pairwiseloss, self).__init__()

    def forward(self, embed, target, chunck_size, Expand=10):
        Expand = np.float64(Expand * 1.0)
        embed_split = torch.split(embed, chunck_size, dim=0)
        target_split = torch.split(target, chunck_size, dim=0)
        Phi = cosine_similarity(embed_split[0], embed_split[1]) * Expand
        soft_phi = softplus(Phi)
        if device == "cuda":
            mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.cuda.FloatTensor)
        else:
            mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.FloatTensor)
        # print(mask.type())
        # print(Phi.type())
        mask_phi = torch.mul(mask, Phi)
        pairwiseloss = torch.sub(soft_phi, mask_phi).mean()
        return pairwiseloss


class pairwiseloss_global(torch.nn.Module):
    def __init__(self):
        super(pairwiseloss_global, self).__init__()

    def forward(self, embed, target, chunk_size, global_center=None, beta=2.0, Expand=10):
        # Expand=10 equals to the temperature=0.1 in a commonly used temperature based formula.
        Expand = np.float64(Expand * 1.0)
        embed_split = torch.split(embed, chunk_size, dim=0)
        target_split = torch.split(target, chunk_size, dim=0)
        Phi = cosine_similarity(embed_split[0], embed_split[1]) * Expand
        soft_phi = softplus(Phi)
        if device == "cuda":
            mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.cuda.FloatTensor)
        else:
            mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.FloatTensor)
        mask_phi = torch.mul(mask, Phi)
        pairwiseloss = torch.sub(soft_phi, mask_phi).mean()
        if global_center is not None:
            if device == "cuda":
                sample_centers = torch.mm(target.type(torch.cuda.FloatTensor), global_center)
            else:
                sample_centers = torch.mm(target.type(torch.FloatTensor), global_center)
            phi_1 = cosine_similarity(embed, sample_centers) * Expand
            global_sim_loss = torch.sub(softplus(phi_1), phi_1).mean()
            # print(pairwiseloss)
            # print(global_sim_loss)
            return pairwiseloss + beta * global_sim_loss
        else:
            return pairwiseloss


class crossentropy_global(torch.nn.Module):
    def __init__(self):
        super(crossentropy_global, self).__init__()

    def forward(self, output, target, embed, chuck_size, global_center=None, beta=0.25, Expand=10):
        # cross entropy loss
        # print(target.size())
        _, target_cce = target.max(1)
        # print(target_cce.size())
        # sys.exit()
        cross_loss = cross_entropy(output, target_cce)

        if global_center is not None:
            # print("hello=================")
            # global_center.to(device)
            Expand = np.float64(Expand * 1.0)
            sample_centers = torch.mm(target.type(torch.float64), global_center)
            phi_1 = cosine_similarity(embed, sample_centers) * Expand
            global_sim_loss = torch.sub(softplus(phi_1), phi_1).mean()
            return cross_loss + beta * global_sim_loss
        else:
            return cross_loss


class crossentropy_pairwise(torch.nn.Module):
    def __init__(self):
        super(crossentropy_pairwise, self).__init__()

    def forward(self, output, target, embed, chunck_size, global_center=None, beta=0.25, Expand=10):
        # cross entropy loss
        _, target_cce = target.max(1)
        # print(target_cce.size())
        # sys.exit()
        loss = cross_entropy(output, target_cce)
        # Pairwise loss
        if embed.size()[0] > 4:  # in case of test phase when batch_size is 1 
            Expand = np.float64(Expand * 1.0)
            embed_split = torch.split(embed, chunck_size, dim=0)
            target_split = torch.split(target, chunck_size, dim=0)
            Phi = cosine_similarity(embed_split[0], embed_split[1]) * Expand
            soft_phi = softplus(Phi)
            if device == "cuda":
                mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.cuda.FloatTensor)
            else:
                mask = torch.sum(torch.mul(target_split[0], target_split[1]), dim=1).type(torch.FloatTensor)
            mask_phi = torch.mul(mask, Phi)
            pairwiseloss = torch.sub(soft_phi, mask_phi).mean()

            # print("cross loss: % .3f" % cross_loss)
            # print("pairwise loss: % .3f" % pairwiseloss)
            loss += 0.25 * pairwiseloss
        # Possible global loss
        if global_center is not None:
            # print("hello=================")
            # global_center.to(device)
            Expand = np.float64(Expand * 1.0)
            sample_centers = torch.mm(target.type(torch.float64), global_center)
            phi_1 = cosine_similarity(embed, sample_centers) * Expand
            global_sim_loss = torch.sub(softplus(phi_1), phi_1).mean()
            return loss + beta * global_sim_loss
        else:
            return loss
