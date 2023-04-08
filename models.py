# coding=UTF-8
import numpy as np
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LeakyReLU
from tensorflow.keras.models import Sequential
from tcn.tcn import TCN

class modelss:
    def __init__(self,X_train, X_test, Y_train, Y_test, scaled_tool):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_tedst = Y_test
        self.scaled  = scaled_tool
        self.epochs  = 60
        self.batch   = 32
        self.units   = 20

    def run_tcn_gru(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
        model.add(GRU(units=self.units, return_sequences= False))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_lstm(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
        model.add(LSTM(units=self.units, return_sequences= False))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_rnn(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1,2,4],return_sequences=True))
        model.add(SimpleRNN(units=self.units, return_sequences= False))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_bpnn(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=False))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre


    def run_GRU(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(GRU(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_LSTM(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(LSTM(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_RNN(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(SimpleRNN(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre
    def run_BPNN(self):
        model = Sequential()
        model.add(Dense(self.units, activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(self.X_test)

        Y_pre = Y_pre.reshape(self.X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre