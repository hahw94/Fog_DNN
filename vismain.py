from datasets.vis.util import parameter_grid
from run.visrun import run_model





region = 'temp'
obs_point_list = [887, 921]
for obs_point in obs_point_list:
    data_path = './fogdata/' + str(obs_point)
    save_path = './result/vis/'
    num_unit_1_list = [128] # number of first and second layer of dnn
    num_unit_2_list = [128] # number of thrid layer of dnn
    num_unit_3_list = [32,64] # number of forth layer of dnn
    lr_list = [0.0004] # learning rate
    batch_size_list = [64] # batch size
    epochs_list = [5] # epochs num
    k_fold_list = [5] # num of fold
    vis_limit_list = [10000]

    all_parameters_list = parameter_grid(num_unit_1_list, num_unit_2_list,
                                          num_unit_3_list, lr_list,
                                          batch_size_list, epochs_list,
                                          k_fold_list)
    print("학습 예정 모델 개수 : {}".format(len(all_parameters_list)*len(vis_limit_list)))

    # set each preprocessing data type
    if obs_point == 921:
                pre_data_list = [([], 'stand'),
                             (['AT','RH','WS','WD','Td','WT','WT-AT','bAT','bWS','bWD','bTd'], 'norm'),
                             ([], 'sin'),
                             (['Local Time','DOY','vis','Fog_30'], 'none'),]
    elif obs_point == 887:
        pre_data_list = [([], 'stand'),
                 (['AT','RH','WS','WD','Td','WT','WT-AT'], 'norm'),
                 ([], 'sin'),
                 (['Local Time','DOY','vis','Fog_30'], 'none'),]

    run_model(data_path, save_path, pre_data_list, obs_point, vis_limit_list, all_parameters_list)
