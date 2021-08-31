import time
import torch
import numpy as np
import pandas as pd
from math import sqrt
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import (mean_squared_error,
                             confusion_matrix,
                             classification_report,
                             f1_score,
                             precision_score,
                             recall_score)

from imblearn.over_sampling import SMOTE

from datasets.occur.split_train_test import fog_data_train_test_split
from datasets.occur.util import (convert_torch_type,
                                 binary_acc)

from model.occdnn import Deep_Neural_Network_Target_Existence



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



def return_precision_recall_f1(true_value, predict_value, pos_label = 1):
    precision = precision_score(true_value.squeeze(), predict_value.squeeze(), pos_label=pos_label)
    recall = recall_score(true_value.squeeze(), predict_value.squeeze(), pos_label=pos_label)
    f1_score_value = f1_score(true_value.squeeze(), predict_value.squeeze(), pos_label = pos_label)
    rmse = sqrt(mean_squared_error(true_value.squeeze(), predict_value.squeeze()))
    return precision, recall, f1_score_value, rmse




def run_model(data_path, save_path, pre_data_list, obs_point, vis_limit_list, all_parameters_list):
    for vis_limit_num in vis_limit_list:
        obs_save_path = save_path + str(obs_point) + "\\" + str(vis_limit_num)

        train_x, test_x, train_y, test_y = fog_data_train_test_split(data_path, save_path, obs_point, pre_data_list,
                                                                     vis_limit_num, 'Fog_30', 0.3)

        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        X_train_over, y_train_over = SMOTE(random_state=0).fit_resample(train_x,
                                                                        train_y)  # smote.fit_sample(train_x,train_y)
        print('불균형 데이터 알고리즘 적용 전 학습용 피처/레이블 데이터 세트: ', train_x.shape, train_y.shape)
        print('불균형 데이터 알고리즘 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
        print('불균형 데이터 알고리즘 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

        torch_train_x, torch_train_y, torch_test_x, torch_test_y = convert_torch_type(X_train_over, y_train_over,
                                                                                      test_x, test_y)
        torch_train_x = torch_train_x.reshape([torch_train_x.shape[0], torch_train_x.shape[1]])
        torch_test_x = torch_test_x.reshape([torch_test_x.shape[0], torch_test_x.shape[1]])
        torch_train_y = torch_train_y.reshape([torch_train_y.shape[0], 1])
        torch_test_y = torch_test_y.reshape([torch_test_y.shape[0], 1])
        print(torch_train_x.shape, torch_train_y.shape, torch_test_x.shape, torch_test_y.shape)

        test_nonoccurrence_list = []
        test_occurrence_list = []
        test_occurrence_precision_list = []

        train_nonoccurrence_list = []
        train_occurrence_list = []
        train_occurrence_precision_list = []

        train_rmse_list = []
        test_rmse_list = []

        train_f1_score_list = []
        test_f1_score_list = []

        for i in range(len(all_parameters_list)):
            print("진행률 : {:.2f}%".format(((i + 1) / len(all_parameters_list)) * 100))
            start = time.time()
            num_unit_1, num_unit_2, num_unit_3, lr, batch_size, n_epochs, drop_out = all_parameters_list[i][0], \
                                                                                     all_parameters_list[i][1], \
                                                                                     all_parameters_list[i][2], \
                                                                                     all_parameters_list[i][3], \
                                                                                     all_parameters_list[i][4], \
                                                                                     all_parameters_list[i][5], \
                                                                                     all_parameters_list[i][6],

            all_parameters_list = change_batch_size(vis_limit_num, all_parameters_list, i)

            print(torch_train_x.shape, torch_train_y.shape, torch_test_x.shape, torch_test_y.shape)
            train = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
            test = torch.utils.data.TensorDataset(torch_test_x, torch_test_y)
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
            seed = 1
            lr = lr
            momentum = 0.5
            no_cuda = False
            batch_size = batch_size
            # 배치사이즈와 모멘텀등 몇가지 변수들을 지정해줍니다.
            torch.manual_seed(seed)
            use_cuda = not no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            network = Deep_Neural_Network_Target_Existence(num_unit_1=num_unit_1, num_unit_2=num_unit_2,
                                                           num_unit_3=num_unit_3, input_num=torch_train_x.shape[1],
                                                           drop_out=drop_out).to(device)
            optimizer = optim.Adam(network.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            # 모델구조를 만들어 줍니다.

            for epoch in range(n_epochs):
                epoch_loss = 0
                epoch_acc = 0
                # print('\nEpoch {} / {} \nFold number {} / {}'.format(epoch + 1, epochs, fold + 1 , kfold.get_n_splits()))
                # network.train()
                for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    # print(x_batch.shape)
                    y_pred = network(x_batch)
                    loss = criterion(y_pred, y_batch)
                    # print(y_pred.shape)
                    acc = binary_acc(y_pred, y_batch)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_acc += acc.item()

                # Test Loss
                test_epoch_loss = 0
                test_epoch_acc = 0
                for batch_index, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = network(x_batch)
                    test_loss = criterion(y_pred, y_batch)
                    test_acc = binary_acc(y_pred, y_batch)
                    test_epoch_loss += test_loss.item()
                    test_epoch_acc += test_acc.item()

                # append test,train loss to list
                if (epoch % 100 == 0) | (epoch == n_epochs-1):
                    test_predict = torch.round(torch.sigmoid(
                        network(torch.tensor(torch_test_x, dtype=torch.float).cuda()).clone().detach())).cpu().detach().numpy().squeeze()

                    test_occurrence_accuracy, test_occurrence_precision, test_f1_score, test_rmse = return_precision_recall_f1(test_predict, torch_test_y)

                    test_nonoccurrence_accuracy, test_nonoccurrence_precision, test_nonf1_score, test_rmse = return_precision_recall_f1(test_predict, torch_test_y, 0)

                    print("Train Epoch : {}  Train Loss: {:.6f}  Train Acc: {:.3f}  Test Loss: {:.6f}  Test Acc: {:.3f}".
                        format(epoch, epoch_loss, epoch_acc / len(train_loader), test_epoch_loss,
                               test_epoch_acc / len(test_loader)))

                    print("(TEST) Occurrence Accuracy : {:.2f} / Occurrence Precision : {:.2f} / F1 Score : {} ".
                          format(test_occurrence_accuracy, test_occurrence_precision,  test_f1_score))

                    print("(TEST) Nonoccurrence Accuracy : {:.2f} / Nonoccurrence Precision : {:.2f} / Nonoccurrence F1 Score : {}\n ".
                        format(test_nonoccurrence_accuracy, test_nonoccurrence_precision, test_nonf1_score))


            # test_predict
            test_predict = torch.round(torch.sigmoid(
                network(torch.tensor(torch_test_x, dtype=torch.float).cuda()).clone().detach())).cpu().detach().numpy().squeeze()

            test_occurrence_accuracy, test_occurrence_precision, test_f1_score, test_rmse = return_precision_recall_f1(
                test_predict, torch_test_y)
            test_nonoccurrence_accuracy, test_nonoccurrence_precision, test_nonf1_score, test_rmse = return_precision_recall_f1(
                test_predict, torch_test_y, 0)

            # append test predict to list
            test_nonoccurrence_list.append(test_nonoccurrence_accuracy)
            test_occurrence_list.append(test_occurrence_accuracy)
            test_occurrence_precision_list.append(test_occurrence_precision)
            test_f1_score_list.append(test_f1_score)
            test_rmse_list.append(test_rmse)

            # train predict
            train_predict = torch.round(torch.sigmoid(network(
                torch.tensor(torch.tensor(np.array(train_x), dtype=torch.float32), dtype=torch.float).cuda()).clone().detach())).cpu().detach().numpy().squeeze()
            train_occurrence_accuracy, train_occurrence_precision, train_f1_score, train_rmse = return_precision_recall_f1(
                train_predict, train_y)
            train_nonoccurrence_accuracy, train_nonoccurrence_precision, train_nonf1_score, train_rmse = return_precision_recall_f1(
                train_predict, train_y, 0)

            # append train predict to list
            train_nonoccurrence_list.append(train_nonoccurrence_accuracy)
            train_occurrence_list.append(train_occurrence_accuracy)
            train_occurrence_precision_list.append(train_occurrence_precision)
            train_f1_score_list.append(train_f1_score)
            train_rmse_list.append(train_rmse)

            # save torch model
            torch.save(network.state_dict(), save_path + str(obs_point) + "\\" + str(
                vis_limit_num) + '\\Train_{}_under_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
                obs_point, vis_limit_num, num_unit_1, num_unit_2, num_unit_3, lr, batch_size, n_epochs, drop_out))

        # save model result to csv file
        result_df = pd.DataFrame({'Train Nonoccurrence': train_nonoccurrence_list,
                                  'Train Occurrence': train_occurrence_list,
                                  'Train Occurrence precision': train_occurrence_precision_list,
                                  'Train F1 score': train_f1_score_list,
                                  'Train RMSE': train_rmse_list,
                                  'Test Nonoccurrence': test_nonoccurrence_list,
                                  'Test Occurrence': test_occurrence_list,
                                  'Test Occurrence precision': test_occurrence_precision_list,
                                  'Test F1 score': test_f1_score_list,
                                  'Test RMSE': test_rmse_list,})

        # Save Model parameters Result
        columns = ['layer_1', 'layer_2', 'layer_3', 'lr', 'bs', 'epochs', 'drop_out']
        nn_structure = pd.DataFrame(all_parameters_list, columns=columns)
        result_df = pd.concat([result_df, nn_structure], axis=1)
        sort_result_df = result_df.sort_values('Test RMSE', ascending=True)
        file_info = "{}_{}_all_model_result".format(str(obs_point), str(vis_limit_num)) + ".csv"
        sort_result_df.to_csv(save_path + str(obs_point) + "\\" + str(vis_limit_num) + '\\' + file_info)
