#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2021.11.1
@Author  : Herry
'''
import numpy as np
from vmdpy import VMD
from PyEMD import EMD, EEMD

class Decomposition:
    def __init__(self, data, length):
        self.data = data.reshape(-1)
        self.length = length

    def SSA(self):
        series = self.data
        # step1 嵌入
        windowLen = self.length  # 嵌入窗口长度
        seriesLen = len(series)  # 序列长度
        K = seriesLen - windowLen + 1
        X = np.zeros((windowLen, K))
        for i in range(K):
            X[:, i] = series[i:i + windowLen]
        # step2: svd分解， U和sigma已经按升序排序
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        for i in range(VT.shape[0]):
            VT[i, :] *= sigma[i]
        A = VT

        # 重组
        rec = np.zeros((windowLen, seriesLen))
        for i in range(windowLen):
            for j in range(windowLen - 1):
                for m in range(j + 1):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (j + 1)
            for j in range(windowLen - 1, seriesLen - windowLen + 1):
                for m in range(windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= windowLen
            for j in range(seriesLen - windowLen + 1, seriesLen):
                for m in range(j - seriesLen + windowLen, windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (seriesLen - j)

        return rec.T

    def EMD(self):
        data = self.data
        decomp = EMD()
        decomp.emd(data)
        imfs, res = decomp.get_imfs_and_residue()
        IMFs = imfs.T
        IMFs = np.insert(IMFs, IMFs.shape[1], values=res, axis = 1)

        return IMFs

    def EEMD(self):
        data = self.data
        decomp = EEMD()
        decomp.eemd(data)
        imfs, res = decomp.get_imfs_and_residue()
        IMFs = imfs.T
        IMFs = np.insert(IMFs, IMFs.shape[1], values=res, axis = 1)

        return IMFs

    def VMD(self):
        data = self.data
        alpha, tau, length, DC, init, tol = 5000, 0, self.length, 0, 1, 1e-8
        u, u_hat, omega = VMD(data, alpha, tau, length, DC, init, tol)

        return u.T
