import glob
import numpy as np
import pandas as pd


def import_data(data_path, obs_point):
    files = glob.glob(data_path + '\\*.csv')
    leap_month_list = [31,29,31,30,31,30,31,31,30,31,30,31]
    normal_month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
    leap_year_list = []
    normal_year_list = []
    for i in range(sum(leap_month_list)):
        temp = [i+1]*144
        leap_year_list += temp

    for i in range(sum(normal_month_list)):
        temp = [i+1]*144
        normal_year_list += temp

    leap_doy = np.sin((np.array(leap_year_list)*(2*np.pi))/366)
    normal_doy = np.sin((np.array(normal_year_list)*(2*np.pi))/365)

    for i in range(len(files)):
        if i == 0:
            total_pd = pd.read_csv(files[i])

            # 로컬타임 사인으로 변환한후 더하기
            local_time = total_pd['hour'] + total_pd['minute']/60
            sin_local_time = (np.sin((local_time*2*np.pi)/24))
            total_pd['Local Time'] = sin_local_time

            if total_pd['year'][0] % 4 == 0:
                total_pd['DOY'] = leap_doy
            else:
                total_pd['DOY'] = normal_doy



        else:
            temp_pd = pd.read_csv(files[i])

            # 로컬타임 사인으로 변환한후 더하기
            local_time = temp_pd['hour'] + temp_pd['minute']/60
            sin_local_time = (np.sin((local_time*2*np.pi)/24))
            temp_pd['Local Time'] = sin_local_time

            if temp_pd['year'][0] % 4 == 0:
                temp_pd['DOY'] = leap_doy
            else:
                temp_pd['DOY'] = normal_doy


            total_pd = pd.concat([total_pd, temp_pd], axis = 0)
    return total_pd



def transform_data(df, pre_data, method):
    if method == "stand":
        temp_df = (df[pre_data] - df[pre_data].mean()) / df[pre_data].std()
        return temp_df
    elif method == "norm":
        temp_df = (df[pre_data] - df[pre_data].min()) / (df[pre_data].max() - df[pre_data].min())
        return temp_df
    elif method == "sin":
        temp_df = (np.sin((df[pre_data]*2*np.pi)/df[pre_data].max()))
        return temp_df
    elif method == "none":
        temp_df = df[pre_data]
        return temp_df
    else:
        return print(method, " -> method is wrong value")