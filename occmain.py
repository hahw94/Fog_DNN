import warnings
warnings.filterwarnings('ignore')

from run.occrun import run_model
from datasets.occur.util import (parameter_grid,
                                 all_pre_data_list)
from datasets.occur.split_train_test import fog_data_train_test_split


region = 'temp'
obs_point_list = [887, 921]
for obs_point in obs_point_list:
    data_path = './fogdata/' + str(obs_point)
    save_path = './result/occur/'
    # Parameter values to test
    num_unit_1_list = [128, 256, 512]  # number of first layer of dnn
    num_unit_2_list = [128, 256, 512]  # number of second layer of dnn
    num_unit_3_list = [32, 64, 128]  # number of third layer of dnn
    lr_list = [0.0004]  # learning rate
    batch_size_list = [64]  # batch size
    epochs_list = [500, 1000]  # epochs num
    drop_out_list = [0]  # drop_out_rate
    vis_limit_list = [1000] # visbility limit value

    all_parameters_list = parameter_grid(num_unit_1_list, num_unit_2_list,
                                          num_unit_3_list, lr_list,
                                          batch_size_list, epochs_list,
                                          drop_out_list)
    print("학습 예정 모델 개수 : {}".format(len(all_parameters_list) * len(vis_limit_list)))
    if obs_point == 921:
        pre_data_list = [([], 'stand'),
                         (['AT', 'RH', 'WS', 'WD', 'Td', 'WT', 'WT-AT', 'bAT', 'bWS', 'bWD', 'bTd'], 'norm'),
                         ([], 'sin'),
                         (['Local Time', 'DOY', 'vis', 'Fog_30'], 'none'), ]
    elif obs_point == 887:
        pre_data_list = [([], 'stand'),
                         (['AT', 'RH', 'WS', 'WD', 'Td', 'WT', 'WT-AT'], 'norm'),
                         ([], 'sin'),
                         (['Local Time', 'DOY', 'vis', 'Fog_30'], 'none'), ]

    input_model_num = len(all_pre_data_list(pre_data_list)) - 2

    for vis_limit_num in vis_limit_list:
        obs_save_path = save_path + str(obs_point) + "\\" + str(vis_limit_num)

        train_x, test_x, train_y, test_y = fog_data_train_test_split(data_path, save_path, obs_point, pre_data_list,
                                                                     vis_limit_num, 'Fog_30', 0.3)

        run_model(data_path, save_path, pre_data_list, obs_point, vis_limit_list, all_parameters_list)