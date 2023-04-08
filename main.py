# coding=UTF-8
'''
@Date    : 2022.05.29
@Author  : Jethro
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models import modelss
from Decomposition import Decomposition
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import warnings
warnings.filterwarnings('ignore')
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def denoise(data, imfs):
    data = data.reshape(-1)
    denoise_data = 0
    for i in range(imfs.shape[1]):
        denoise_data += imfs[:,i]
        pearson_corr_coef = np.corrcoef(denoise_data, data)
        if pearson_corr_coef[0,1] >=0.995:
            print(i)
            break

    return denoise_data

def IMF_decomposition(data, length):
    Decomp = Decomposition(data, length)
    # emd_imfs  = Decomp.EMD()
    # eemd_imfs = Decomp.EEMD()
    # vmd_imfs  = Decomp.VMD()
    ssa_imfs  = Decomp.SSA()
    # emd_denoise = denoise(data, emd_imfs)
    # eemd_denoise = denoise(data, eemd_imfs)
    # vmd_denoise = denoise(data, vmd_imfs)
    ssa_denoise = denoise(data, ssa_imfs)
    # emd_denoise, eemd_denoise, vmd_denoise,

    return ssa_denoise

def Data_partitioning(data,test_number, input_step, pre_step):
    # 导入数据
    dataset = data.reshape(-1,1)
    test_number = test_number
    # #归一化
    scaled_tool = MinMaxScaler(feature_range=[0, 1])
    data_scaled = scaled_tool.fit_transform(dataset)
    # 切片
    step_size = input_step
    data_input= np.zeros((len(data_scaled) - step_size - pre_step, step_size))
    data_label = np.zeros((len(data_scaled) - step_size - pre_step, 1))
    for i in range(len(data_scaled) - step_size-pre_step):
        data_input[i, :] = data_scaled[i:step_size + i,0]
        data_label[i, 0] = data_scaled[step_size + i + pre_step,0]
    # data_label = data_scaled[step_size+1:,0]
    # 划分数据集
    X_train = data_input[:-test_number]
    Y_train = data_label[:-test_number]
    X_test = data_input[-test_number:]
    Y_test = data_label[-test_number:]

    return  X_train, X_test, Y_train, Y_test, scaled_tool

def single_model(data,test_number,flag, input_step, pre_step):
    X_train, X_test, Y_train, Y_test, scaled_tool = Data_partitioning(data, test_number, input_step, pre_step)
    model = modelss(X_train, X_test, Y_train, Y_test, scaled_tool)
    if flag == 'tcn_gru':
        pre = model.run_tcn_gru()
    if flag == 'tcn_lstm':
        pre = model.run_tcn_lstm()
    if flag == 'tcn_rnn':
        pre = model.run_tcn_rnn()
    if flag == 'tcn_bpnn':
        pre = model.run_tcn_bpnn()
    if flag == 'gru':
        pre = model.run_GRU()
    if flag == 'lstm':
        pre = model.run_LSTM()
    if flag == 'rnn':
        pre = model.run_RNN()
    if flag == 'bpnn':
        pre = model.run_BPNN()
    data_pre = pre[:, 0]

    return data_pre


if __name__ == '__main__':
    test_number, imfs_number, input_step, pre_step= 200, 15, 20, 2
    data = pd.read_csv('10 min wind speed data.csv', header= None)
    ssa_denoise = IMF_decomposition(data.iloc[:,2].values, imfs_number)
    np.savetxt('ssa_denoise_3.csv',ssa_denoise[-test_number:],delimiter = ',')
    # pre_emd = single_model(emd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_eemd = single_model(eemd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_vmd = single_model(vmd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    pre_ssa_tcn_gru = single_model(ssa_denoise, test_number, 'tcn_gru', input_step, pre_step)
    pre_ssa_tcn_lstm = single_model(ssa_denoise, test_number, 'tcn_lstm', input_step, pre_step)
    pre_ssa_tcn_rnn = single_model(ssa_denoise, test_number, 'tcn_rnn', input_step, pre_step)
    pre_ssa_tcn_bpnn = single_model(ssa_denoise, test_number, 'tcn_bpnn', input_step, pre_step)
    pre_ssa_gru = single_model(ssa_denoise, test_number, 'gru', input_step, pre_step)
    pre_ssa_lstm = single_model(ssa_denoise, test_number, 'lstm', input_step, pre_step)
    pre_ssa_rnn = single_model(ssa_denoise, test_number, 'rnn', input_step, pre_step)
    pre_ssa_bpnn = single_model(ssa_denoise, test_number, 'bpnn', input_step, pre_step)
    # np.savetxt('pre_emd.csv', pre_emd, delimiter=',')
    # np.savetxt('pre_eemd.csv', pre_eemd, delimiter=',')
    # np.savetxt('pre_vmd.csv', pre_vmd, delimiter=',')
    np.savetxt('pre_ssa_tcn_gru.csv', pre_ssa_tcn_gru, delimiter=',')
    np.savetxt('pre_ssa_tcn_lstm.csv', pre_ssa_tcn_lstm, delimiter=',')
    np.savetxt('pre_ssa_tcn_rnn.csv', pre_ssa_tcn_rnn, delimiter=',')
    np.savetxt('pre_ssa_tcn_bpnn.csv', pre_ssa_tcn_bpnn, delimiter=',')
    np.savetxt('pre_ssa_gru.csv', pre_ssa_gru, delimiter=',')
    np.savetxt('pre_ssa_lstm.csv', pre_ssa_lstm, delimiter=',')
    np.savetxt('pre_ssa_rnn.csv', pre_ssa_rnn, delimiter=',')
    np.savetxt('pre_ssa_bpnn.csv', pre_ssa_bpnn, delimiter=',')
    Actual = ssa_denoise[-test_number:]

    print('实验一：不同分解方法对预测结果影响')
    # print('#########################')
    # print('EMD分解方法： MAE : ', mae(Actual, pre_emd))
    # print('EMD分解方法： R2 : ', mape(Actual, pre_emd))
    # print('EMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_emd)))
    # print('#########################')
    # print('EEMD分解方法： MAE : ', mae(Actual, pre_eemd))
    # print('EEMD分解方法： R2 : ', mape(Actual, pre_eemd))
    # print('EEMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_eemd)))
    # print('#########################')
    # print('VMD分解方法： MAE : ', mae(Actual, pre_vmd))
    # print('VMD分解方法： R2 : ', mape(Actual, pre_vmd))
    # print('VMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_vmd)))
    # print('#########################')
    print('SSA分解方法： MAE : ', mae(Actual, pre_ssa_tcn_gru))
    print('SSA分解方法： R2 : ', mape(Actual, pre_ssa_tcn_gru))
    print('SSA分解方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_gru)))

    print('实验二：采用不同时间信息提取模型对实验结果影响')
    print('#########################')
    print('TCN-LSTM方法： MAE : ', mae(Actual, pre_ssa_tcn_lstm))
    print('TCN-LSTM方法： R2 : ', mape(Actual, pre_ssa_tcn_lstm))
    print('TCN-LSTM方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_lstm)))
    print('#########################')
    print('TCN-RNN方法： MAE : ', mae(Actual, pre_ssa_tcn_rnn))
    print('TCN-RNN方法： R2 : ', mape(Actual, pre_ssa_tcn_rnn))
    print('TCN-RNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_rnn)))
    print('#########################')
    print('TCN-BPNN方法： MAE : ', mae(Actual, pre_ssa_tcn_bpnn))
    print('TCN-BPNN方法： R2 : ', mape(Actual, pre_ssa_tcn_bpnn))
    print('TCN-BPNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_bpnn)))

    print('实验三：与基模性对比')
    print('#########################')
    print('GRU方法： MAE : ', mae(Actual, pre_ssa_gru))
    print('GRU方法方法： R2 : ', mape(Actual, pre_ssa_tcn_lstm))
    print('GRU方法方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_lstm)))
    print('#########################')
    print('LSTM方法： MAE : ', mae(Actual, pre_ssa_lstm))
    print('LSTM方法： R2 : ', mape(Actual, pre_ssa_lstm))
    print('LSTM方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_lstm)))
    print('#########################')
    print('RNN方法： MAE : ', mae(Actual, pre_ssa_rnn))
    print('RNN方法： R2 : ', mape(Actual, pre_ssa_rnn))
    print('RNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_rnn)))
    print('#########################')
    print('BPNN方法： MAE : ', mae(Actual, pre_ssa_bpnn))
    print('BPNN方法： R2 : ', mape(Actual, pre_ssa_bpnn))
    print('BPNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_bpnn)))


    plt.figure(2)
    plt.plot(pre_ssa_tcn_gru, color = 'black', label= 'pre_ssa_tcn_gru')
    plt.plot(pre_ssa_tcn_lstm, color= 'm', label= 'pre_ssa_tcn_lstm')
    plt.plot(pre_ssa_tcn_rnn, color= 'y', label= 'pre_ssa_tcn_rnn')
    plt.plot(pre_ssa_tcn_bpnn, color = 'red', label= 'pre_ssa_tcn_bpnn')
    plt.plot(pre_ssa_gru, color= 'y', label= 'pre_ssa_gru')
    plt.plot(pre_ssa_lstm, color = 'red', label= 'pre_ssa_lstm')
    plt.plot(Actual, color= 'blue', label= 'Actual')
    plt.legend()
    plt.show()
