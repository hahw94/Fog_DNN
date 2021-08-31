import os
import torch
from itertools import product
import numpy as np

from datasets.occur.preprocess import import_data


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def convert_torch_type(input_train_data, target_train_data, input_test_data, target_test_data):
    torch_train_x = torch.tensor(np.array(input_train_data), dtype=torch.float32)
    torch_train_y = torch.tensor(np.array(target_train_data), dtype=torch.float32)
    torch_test_x = torch.tensor(np.array(input_test_data), dtype=torch.float32)
    torch_test_y = torch.tensor(np.array(target_test_data), dtype=torch.float32)
    return torch_train_x, torch_train_y, torch_test_x, torch_test_y


def using_data_name_list(pre_data_list):
    total_list = []
    for i in range(len(pre_data_list)):
        total_list += pre_data_list[i][0]
    return total_list


def parameter_grid(num_unit_1_list, num_unit_2_list, num_unit_3_list, lr_list, batch_size_list, epochs_list, k_fold_list, ):
    items = [num_unit_1_list, num_unit_2_list, num_unit_3_list, lr_list, batch_size_list, epochs_list, k_fold_list, ]
    result = list(product(*items))
    return result


def print_data_columns(data_path, obs_point):
    total_df = import_data(data_path, obs_point)
    return total_df.columns


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def all_pre_data_list(pre_data_list):
    total_list = []
    for i in range(len(pre_data_list)):
        total_list += pre_data_list[i][0]
    return total_list



