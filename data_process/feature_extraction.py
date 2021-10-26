"""
1. Feature extraction.
    1. amplitude
    2. FFT
2. Action(label) encoding
    1. str -> int -> one-hot
"""
import numpy as np
import os
import _pickle as pickle
from tqdm import tqdm
import argparse


action_dict = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}  # global activity encode dict.


def load_pickle(filename, show_name=False):
    if show_name:
        print(filename)
    return pickle.load(open(filename, "rb"))


def save_pickle(data_dict, filename):
    with open(filename, "wb") as myfile:
        pickle.dump(data_dict, myfile, -1)


def axisData_split(sensor_data, interval_len):
    """
    split sensor data into small interval, for LSTM unit training
    sensor_data: numpy.array with shape [seq_len, 8]
    interval_len: int, the length of the resulting small interval
    :return: list of numpy.array with shape [seq_len/interval_len, interval_len. 8]
    """
    assert type(interval_len) == int
    result = []
    # print(len(sensor_data[0]) / tao)
    start, end = 0, interval_len
    while end <= len(sensor_data):
        result.append(sensor_data[start:end])
        start = end
        end += interval_len
    return result


def dim_expansion(data):
    """
    add the fourth dim for the original
    :param data: numpy.array ax, ay, az, mx, my, mz data with shape [interval_len, 6]
    :return: numpy.array with shape [interval_len, 8]
    """
    data_new = np.zeros((data.shape[0], 8), dtype=float)
    for i in range(data.shape[0]):
        original_axis_index = 0
        for j in range(8):
            if j != 3 and j != 7:
                data_new[i][j] = data[i][original_axis_index]
                original_axis_index += 1
            else:
                data_new[i][j] = np.sqrt(
                    np.power(data_new[i][j - 1], 2) + np.power(data_new[i][j - 2], 2) + np.power(data_new[i][j - 3],
                                                                                                 2))
    return data_new


def feature_extraction_dict(data_np, dim=4, tao=0.5, freq=50):
    """Feature extraction for each data sample, FFT, frequency and magnitude"""
    if data_np.shape[0] < data_np.shape[1]:
        data_np = np.rollaxis(data_np, 1, 0)
    if dim == 4:
        data_expand = dim_expansion(data_np)    # calculate the amplitude
    else:
        data_expand = data_np

    # split according to time interval: e.g. tao = 0.5 seconds
    # small_interval_len = int(freq * tao)  # [lstm_len, interval_len, 8]
    # small_interval_len = int(freq * tao)

    # split with a fixed length of signal.
    small_interval_len = 12
    axis_split_result = axisData_split(sensor_data=data_expand, interval_len=small_interval_len)
    # feature extraction                  # [lstm_len. interval_len/2, 16]
    sample_feature = []
    for lstm_unit in axis_split_result:  #
        unit_feature = []
        for i in range(lstm_unit.shape[-1]):
            # fft_result = np.fft.fft(lstm_unit[:, i]) / len(lstm_unit)
            fft_result = np.fft.fft(lstm_unit[:, i])
            amplitutde = np.sqrt(np.power(fft_result.real, 2) + np.power(fft_result.imag, 2))
            amplitutde = amplitutde[0:int(len(lstm_unit) / 2) + 1]
            freq_val = np.array([val * float(freq) / float(len(lstm_unit)) for val in
                                 range(int(len(lstm_unit) / 2) + 1)])
            unit_feature.append(freq_val)
            unit_feature.append(amplitutde)
        sample_feature.append(np.array(unit_feature))
    return np.array(sample_feature)


def action_encoding(act, one_hot=False):
    act_int = action_dict[act]
    if one_hot:
        result = np.zeros(len(action_dict), dtype=int)
        result[act_int] = 1
        return result
    else:
        return act_int
    

def data_process_sensor(data_str, length=150):
    data_list = data_str.strip().split(",")
    acc_x = []
    acc_y = []
    acc_z = []
    for i in range(int(len(data_list) / 3)):
        acc_x.append(float(data_list[i * 3]))
        acc_y.append(float(data_list[i * 3 + 1]))
        acc_z.append(float(data_list[i * 3 + 2]))
    acc_x = np.array(acc_x[0:length])
    acc_y = np.array(acc_y[0:length])
    acc_z = np.array(acc_z[0:length])
    return np.stack([acc_x, acc_y, acc_z], axis=0)    
    
    
def feature_extract(in_dir, out_dir, one_hot=False):
    """
    Args:
        in_dir: folder where the original txt data stores (format of the txt file can be found in readme)
        out_dir: output .pickle files
        one_hot: where to use one-hot encoding for the activity label
    """
    act_num = {}
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    files = [val for val in os.listdir(in_dir) if 'txt' in val]
    for file in files:
        user = file.split("_")[0]
        result = {"label": [], "feature": [], "label_t": []}           # result for all samples stored in this file
        local_act_sets = set()
        data = open(os.path.join(in_dir, file))
        for line in data:
            uid, act, acc, gyro = line.strip().split("\t")
            # action
            act_enc = action_encoding(act, one_hot=one_hot)
            # sensor signal, [2 * 3, length] (length = 150)
            signal = np.concatenate((data_process_sensor(acc), data_process_sensor(gyro)), axis=0)
            # frequency based on the 7 seconds of acc and gyro signals.
            acc_freq = (len(acc.split(",")) + len(gyro.split(","))) / 14.0  # 7's
            # feature
            result["feature"].append(feature_extraction_dict(signal, dim=4, tao=0.5, freq=acc_freq))  # FFT
            result["label"].append(act_enc)
            local_act_sets.add(action_dict[act])
        # get all local activities.
        user_local_label = np.ones(7) * -1
        local_act_sets = list(local_act_sets)
        local_act_sets.sort()
        for ind, val in enumerate(local_act_sets):
            user_local_label[val] = ind
        act_num[user] = user_local_label
        result["label_t"] = user_local_label[np.array(np.argmax(result["label"], axis=1))]
        if any(result["label_t"] < 0):
            print("local label error.")
        save_pickle(result, os.path.join(out_dir, file.split(".")[0] + ".pickle"))
    save_pickle(act_num, "F:\\www21\\final_version\\Meta-HAR\\Data\\trans_dict_collect.pickle")
    print("Done !")


# def act_new_label(in_dir, out_dir, act_num):
#     file_list = [file for file in os.listdir(in_dir)]
#     for file in file_list:
#         user_id = file.split("_")[0] + "_act"
#         file_name = os.path.join(in_dir, file)
#         data = load_pickle(file_name)
#         data["label_t"] = act_num[user_id][np.array(np.argmax(data["label"], axis=1))]
#         save_pickle(data, os.path.join(out_dir, file))
#
#
# def local_target_label():
#     act_num = load_pickle("F:\\www21\\data\\trans_dict_collect.pickle")
#     collect_dir = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/all_data/collect"
#     collect_out = "/data/ceph/seqrec/fl_data/www21/data/feature_fft/all_data//collect_new"
#     act_new_label(collect_dir, collect_out, act_num)


def main(in_dir, out_dir):
    feature_extract(in_dir, out_dir, one_hot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Processing of the collected HAR dataset')
    # Path Arguments  gender_method
    parser.add_argument('--in_dir', type=str, default="F:\\www21\\final_version\\Meta-HAR\\Data\\collected",
                        help='dir that stores the original txt files')
    parser.add_argument('--out_dir', type=str, default="F:\\www21\\final_version\\Meta-HAR\\Data\\collected_pickle",
                        help='dir that is used to store the processed pickle files')
    params = parser.parse_args()
    main(params.in_dir, params.out_dir)
