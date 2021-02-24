import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import sys


class merge_sensor_block(nn.Module):

    def __init__(self, in_planes, planes):
        super(merge_sensor_block, self).__init__()
        # Between freq and magnitute
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 2, 3), stride=(1, 2, 1), padding=(0, 0, 1))
        self.bn1 = nn.BatchNorm3d(planes)
        # Between sensor axis
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 4, 3), stride=(1, 4, 1), padding=(0, 0, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        # between sensor data
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 2, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class norm_sensor_block(nn.Module):

    def __init__(self, in_planes, planes):
        super(norm_sensor_block, self).__init__()
        # Acc
        self.acc_conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 2, 3), stride=(1, 2, 1), padding=(0, 0, 1))
        self.acc_bn1 = nn.BatchNorm3d(planes)
        # Between sensor axis
        self.acc_conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 4, 3), stride=(1, 4, 1), padding=(0, 0, 1))
        self.acc_bn2 = nn.BatchNorm3d(planes)

        # gyro
        self.gyro_conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 2, 3), stride=(1, 2, 1), padding=(0, 0, 1))
        self.gyro_bn1 = nn.BatchNorm3d(planes)
        # Between sensor axis
        self.gyro_conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 4, 3), stride=(1, 4, 1), padding=(0, 0, 1))
        self.gyro_bn2 = nn.BatchNorm3d(planes)
        # merge conv
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 2, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        """ input x with shape b, c = 1, seq_len, sensor_dim = 16, interval_len = (7 /12 ?)"""
        # is it possible to paralle the acc and gyro convolution ??? TO DO
        # split x
        x_split = torch.split(x, 8, dim=3)
        # acc
        acc_out = F.relu(self.acc_bn1(self.acc_conv1(x_split[0])))
        acc_out = F.relu(self.acc_bn2(self.acc_conv2(acc_out)))
        # gyrp
        gyro_out = F.relu(self.gyro_bn1(self.gyro_conv1(x_split[1])))
        gyro_out = F.relu(self.gyro_bn2(self.gyro_conv2(gyro_out)))

        sensor_data = torch.cat([acc_out, gyro_out], 3)
        out = F.relu(self.bn3(self.conv3(sensor_data)))
        return out


class main_model(nn.Module):
    def __init__(self, sensor_block, seq_len, batch_size, logits_len=None, embed=False, embed_len=100, rnn_layer=2,
                 bidirectional=False):
        super(main_model, self).__init__()
        self.embed = embed
        self.seq_len = seq_len
        self.batch_size = batch_size
        if bidirectional:
            bid = 2
        else:
            bid = 1
        #
        self.conv = nn.Sequential(sensor_block(1, 64))  # [batch, planes, seq_len, 1, 4]
        self.lstm = nn.LSTM(64 * 1 * 4, embed_len, rnn_layer, bidirectional=bidirectional)
        if not embed:
            self.linear = nn.Linear(embed_len * bid, logits_len)

    def forward(self, x):
        """ x with shape [batch, channel, seq_len, H, W] """
        out = self.conv(x)  # out with shape
        # input of rnn must be: seq_len, batch, input_size
        # print(out.size())
        # print(out.permute(2, 0, 1, 3, 4).size())
        # out = out.permute(2, 0, 1, 3, 4)
        # print(out.contiguous().view(self.seq_len, self.batch_size, -1).size())
        # sys.exit()
        out, _ = self.lstm(out.permute(2, 0, 1, 3, 4).contiguous().view(self.seq_len, -1, 64 * 1 * 4))
        # out with shape (seq_len, batch, num_directions * hidden_size)
        # print(out.size())
        out = torch.mean(out.permute(1, 0, 2), 1)  # [batch, num_directions * hidden_size]
        # print(out.size())
        # sys.exit()
        if self.embed:
            return out
        else:
            logits = self.linear(out)
            return logits, out


class last_layer(nn.Module):
    def __init__(self, embed_len, number_class):
        super(last_layer, self).__init__()
        self.number_class = number_class
        self.linear = nn.Linear(embed_len, number_class)

    def forward(self, x):
        return self.linear(x)   # logits


def merge_embed(bidirectional):
    return main_model(sensor_block=merge_sensor_block, seq_len=utils.parameter["SEQUENCE_LEN"],
                      batch_size=utils.parameter["BATCH_SIZE"],
                      embed=True, bidirectional=bidirectional)


def merge_cce(bidirectional, num_classes=6):
    return main_model(sensor_block=merge_sensor_block, seq_len=utils.parameter["SEQUENCE_LEN"],
                      batch_size=utils.parameter["BATCH_SIZE"],
                      logits_len=num_classes, bidirectional=bidirectional)


def norm_embed(bidirectional):
    return main_model(sensor_block=norm_sensor_block, seq_len=utils.parameter["SEQUENCE_LEN"],
                      batch_size=utils.parameter["BATCH_SIZE"],
                      embed=True, bidirectional=bidirectional)


def norm_cce(bidirectional, num_classes=6):
    return main_model(sensor_block=norm_sensor_block, seq_len=utils.parameter["SEQUENCE_LEN"],
                      batch_size=utils.parameter["BATCH_SIZE"],
                      logits_len=num_classes, bidirectional=bidirectional)
