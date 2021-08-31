import os
import torch
from itertools import product
import pandas as pd
import numpy as np

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



def parameter_grid(num_unit_1_list, num_unit_2_list, num_unit_3_list, lr_list, batch_size_list, epochs_list, k_fold_list,):
    items = [num_unit_1_list, num_unit_2_list, num_unit_3_list, lr_list, batch_size_list, epochs_list, k_fold_list,]
    result = list(product(*items))
    return result



def return_corr(model, torch_test_x, torch_test_y):
    r2_pred = model(torch_test_x.cuda()).cpu().detach().numpy()
    r2_target = np.array(torch_test_y)
    df_r2 = pd.DataFrame()
    df_r2['r2_pred'] = r2_pred[:,0]
    df_r2['r2_target'] = r2_target
    return df_r2.corr().iloc[0,1]