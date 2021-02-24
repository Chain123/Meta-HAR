"""
1. Feature extraction.
    1. mag
    2. FFT 
    
2. Action(label) encoding
    1. str -> int -> one-hot
    
"""
import numpy as np
import random
import os
import _pickle as pickle


def load_pickle(filename, show_name=False):
    if show_name:
        print(filename)
    return pickle.load(open(filename, "rb"))


def save_pickle(data_dict, filename):
    with open(filename, "wb") as myfile:
        pickle.dump(data_dict, myfile, -1)


action_dict = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}


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
    data_feature = []
    if data_np.shape[0] < data_np.shape[1]:
        data_np = np.rollaxis(data_np, 1, 0)
    if dim == 4:
        data_expand = dim_expansion(data_np)
        # print(val_expand.shape)
        # sys.exit()
    else:
        data_expand = data_np
    # small_interval_len = int(freq * tao)  # [lstm_len, interval_len, 8]
    #     small_interval_len = int(freq * tao)
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
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)    
    filelist = [val for val in os.listdir(in_dir)]
    for file in filelist:
        result = {"label": [], "feature": []}    # result for all samples stored in this file
        data = open(os.path.join(in_dir, file))
        for line in data:
            uid, act, acc, gyro = line.strip().split("\t") 
            # action
            act_enc = action_encoding(act, one_hot=one_hot)
            # sensor singal, [2 * 3, length] (length = 150)
            signal = np.concatenate((data_process_sensor(acc), data_process_sensor(gyro)), axis=0)
            # frequency
            acc_freq = (len(acc.split(",")) + len(gyro.split(","))) / 14.0  # 7's 
            # feature
            result["feature"].append(feature_extraction_dict(signal, dim=4, tao=0.5, freq=acc_freq))
            result["label"].append(act_enc)
        
        save_pickle(result, os.path.join(out_dir, file.split(".")[0] + ".pickle"))
    print("Done !")


def main()
    in_dir = "original-data-dir"
    out_dir = "(target)feature-dir"
    feature_extract(in_dir, out_dir, one_hot=True)


if __name__ == '__main__':
    main()
