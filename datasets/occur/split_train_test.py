import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.occur.util import (createFolder, using_data_name_list)

from datasets.occur.preprocess import (import_data,
                                       transform_data)

def save_train_test_data(obs_save_path, train_x, test_x, train_y, test_y):
    train_x.to_csv(obs_save_path+"\\"+"train_x.csv", index = False)
    test_x.to_csv(obs_save_path+"\\"+"test_x.csv", index = False)
    train_y.to_csv(obs_save_path+"\\"+"train_y.csv", index = False)
    test_y.to_csv(obs_save_path+"\\"+"test_y.csv", index = False)
    return print("All Data Save to '{}'".format(obs_save_path))


def fog_data_train_test_split(data_path, save_path, obs_point, pre_data_list, vis_limit_num, target_columns, test_size):
    obs_save_path = save_path + str(obs_point) + "\\" + str(vis_limit_num)
    createFolder(obs_save_path)
    files = glob.glob(data_path + '\\*.csv')
    total_df = import_data(data_path, obs_point)
    total_df.replace(-999, np.nan, inplace=True)
    total_pre_data_list = using_data_name_list(pre_data_list)
    remove_nan_data = total_df[total_pre_data_list].dropna()

    for pre_data_info in pre_data_list:
        data_list, preprocessor_method = pre_data_info
        if len(data_list) == 0:
            print(preprocessor_method, ": No data")
        else:
            for data in data_list:

                try:
                    preprocessed_data = pd.concat(
                        [preprocessed_data, transform_data(remove_nan_data, data, preprocessor_method)], axis=1)

                except:
                    preprocessed_data = transform_data(remove_nan_data, data, preprocessor_method)

    if vis_limit_num == 0:
        preprocessed_data_vis = preprocessed_data[preprocessed_data['Fog_30'] == 1]
    else:
        preprocessed_data_vis = preprocessed_data[preprocessed_data['vis'] < vis_limit_num]

    input_data = preprocessed_data_vis[preprocessed_data_vis.columns.difference([target_columns, 'vis'])]
    train_x, test_x, train_y, test_y = train_test_split(input_data,
                                                        (preprocessed_data_vis[target_columns]).astype(np.int),
                                                        test_size=test_size, random_state=42)
    save_train_test_data(obs_save_path, train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y