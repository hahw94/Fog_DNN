import time
import torch
import numpy as np
import pandas as pd
from math import sqrt
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from datasets.vis.split_train_test import fog_data_train_test_split
from datasets.vis.util import (convert_torch_type,
                               return_corr)

from model.visdnn import Deep_Neural_Network



def train(model, device, n_epochs, optimizer, loss_fn, batch_size, kfold, torch_train_x, torch_train_y, torch_test_x, torch_test_y, save_Model=False):
    for fold, (train_index, test_index) in enumerate(kfold.split(torch_train_x, torch_train_y)):
        x_train_fold = torch_train_x[train_index]
        y_train_fold = torch_train_y[train_index]

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                y_batch = y_batch.reshape([y_batch.shape[0],1])
                loss = loss_fn(out, y_batch)
                loss.backward()
                optimizer.step()

            train_r2_score = r2_score(np.array(torch_train_y), model(torch_train_x.cuda()).cpu().detach().numpy())
            train_r_score = return_corr(model, torch_train_x, torch_train_y)
            # train_df = pd.DataFrame({'Target':np.array(torch_train_y),'Model':model(torch_train_x.cuda()).cpu().detach().numpy().reshape(-1)})
            train_rmse = sqrt(mean_squared_error(np.array(torch_train_y), model(torch_train_x.cuda()).cpu().detach().numpy()))

            test_r2_score = r2_score(np.array(torch_test_y), model(torch_test_x.cuda()).cpu().detach().numpy())
            test_r_score = return_corr(model, torch_test_x, torch_test_y)
            # test_df = pd.DataFrame({'Target':np.array(torch_test_y),'Model':model(torch_test_x.cuda()).cpu().detach().numpy().reshape(-1)})
            test_rmse = sqrt(mean_squared_error(np.array(torch_test_y), model(torch_test_x.cuda()).cpu().detach().numpy()))

            if epoch == (n_epochs-1):
                print(" Fold : {}, Epoch : {}\ntrain r2 score : {:.3f}, train r score : {:.3f}, train_rmse : {:.2f}\ntest r2 score : {:.3f}, test r score : {:.3f}, test_rmse : {:.2f}\n\n".
                      format(fold + 1, epoch + 1, train_r2_score, train_r_score, train_rmse,test_r2_score, test_r_score, test_rmse))


    print("Fold : {}, Epoch : {}\n train r2 score : {:.3f}, train r score : {:.3f}, train_rmse : {:.2f}\ntest r2 score : {:.3f}, test r score : {:.3f}, test_rmse : {:.2f}\n\n".
          format(fold + 1, epoch + 1, train_r2_score, train_r_score, train_rmse, test_r2_score, test_r_score, test_rmse))

    return model, train_r2_score, train_r_score, train_rmse, test_r2_score, test_r_score, test_rmse



def change_batch_size(vis_limit_num, all_parameters_list, i):
    if vis_limit_num == 1000:
        batch_size = 128
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        print('batch_size : ', batch_size)
        return all_parameters_list

    elif vis_limit_num == 100:
        batch_size = 16
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        return all_parameters_list

    elif vis_limit_num == 500:
        batch_size = 30
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        return all_parameters_list

    elif vis_limit_num == 5000:
        batch_size = 256
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        print('batch_size : ', batch_size)
        return all_parameters_list

    elif vis_limit_num == 10000:
        batch_size = 512
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        print('batch_size : ', batch_size)
        return all_parameters_list

    elif vis_limit_num == 30000:
        batch_size = 768
        temp_list = list(all_parameters_list[i])
        temp_list[4] = batch_size
        all_parameters_list[i] = tuple(temp_list)
        print('batch_size : ', batch_size)
        return all_parameters_list


def run_model(data_path, save_path, pre_data_list, obs_point, vis_limit_list, all_parameters_list):
    for vis_limit_num in vis_limit_list:
        obs_save_path = save_path + str(obs_point) + "\\" + str(vis_limit_num)

        train_x, test_x, train_y, test_y = fog_data_train_test_split(data_path, save_path, obs_point, pre_data_list,
                                                                     vis_limit_num, 'vis', 0.3)
        torch_train_x, torch_train_y, torch_test_x, torch_test_y = convert_torch_type(train_x, train_y, test_x, test_y)

        train_r_result = []
        train_r2_result = []
        train_rmse_result = []
        test_r_result = []
        test_r2_result = []
        test_rmse_result = []
        used_time_list = []

        for i in range(len(all_parameters_list)):
            print("진행률 : {:.2f}%".format(((i + 1) / len(all_parameters_list)) * 100))
            start = time.time()
            num_unit_1, num_unit_2, num_unit_3, lr, batch_size, n_epochs, n_fold = all_parameters_list[i][0], \
                                                                                   all_parameters_list[i][1], \
                                                                                   all_parameters_list[i][2], \
                                                                                   all_parameters_list[i][3], \
                                                                                   all_parameters_list[i][4], \
                                                                                   all_parameters_list[i][5], \
                                                                                   all_parameters_list[i][6],

            all_parameters_list = change_batch_size(vis_limit_num, all_parameters_list, i)

            print(torch_train_x.shape, torch_train_y.shape, torch_test_x.shape, torch_test_y.shape)

            seed = 1
            lr = lr
            momentum = 0.5
            no_cuda = False
            batch_size = batch_size
            # 배치사이즈와 모멘텀등 몇가지 변수들을 지정해줍니다.
            torch.manual_seed(seed)
            use_cuda = not no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            model = Deep_Neural_Network(num_unit_1=num_unit_1, num_unit_2=num_unit_2, num_unit_3=num_unit_3, input_num = torch_train_x.shape[1]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss(reduction='sum')
            kfold = KFold(n_splits=n_fold, shuffle=False)
            network, train_r2_score, train_r_score, train_rmse, test_r2_score, test_r_score, test_rmse = train(model,
                                                                                                               device,
                                                                                                               n_epochs,
                                                                                                               optimizer,
                                                                                                               loss_fn,
                                                                                                               batch_size,
                                                                                                               kfold,
                                                                                                               torch_train_x,
                                                                                                               torch_train_y,
                                                                                                               torch_test_x,
                                                                                                               torch_test_y)

            used_time = time.time() - start
            used_time_list.append(used_time)
            print("현재환경", num_unit_1, num_unit_2, num_unit_3, lr, batch_size, n_epochs, n_fold)
            print("총 걸린시간은 {}초 입니다.\n\n\n\n\n\n".format(used_time))

            train_r_result.append(train_r_score)
            test_r_result.append(test_r_score)
            train_r2_result.append(train_r2_score)
            test_r2_result.append(test_r2_score)
            train_rmse_result.append(train_rmse)
            test_rmse_result.append(test_rmse)

            torch.save(network.state_dict(),
                       obs_save_path + '\\Train_{}_under_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(obs_point, vis_limit_num,
                                                                                             num_unit_1, num_unit_2,
                                                                                             num_unit_3, lr, batch_size,
                                                                                             n_epochs, n_fold))
            try:
                # Save model reuslt
                result_df = pd.DataFrame(
                    {'Train R': train_r_result, 'Train R2': train_r2_result, 'Train RMSE': train_rmse_result,
                     'Test R': test_r_result, 'Test R2': test_r2_result, 'Test RMSE': test_rmse_result})
                columns = ['layer_1', 'layer_2', 'layer_3', 'lr', 'bs', 'epochs', 'kfold']
                nn_structure = pd.DataFrame(all_parameters_list, columns=columns)
                result_df = pd.concat([result_df, nn_structure], axis=1)
                sort_result_df = result_df.sort_values('Test RMSE', ascending=True)
                file_name = "{}_{}_all_model_result".format(str(obs_point), str(vis_limit_num)) + ".csv"
                obs_save_path = save_path + str(obs_point) + "\\" + str(vis_limit_num)
                sort_result_df.to_csv(obs_save_path + "\\" + file_name)
            except:
                print("Result csv file is open !")
    return None