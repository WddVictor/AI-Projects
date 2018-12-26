# _*_ coding: utf-8 _*_

import numpy as np
import random
import argparse
import os
from datetime import datetime
import sys
import re

instance_file = './datasets/train_data.txt'
test_file_path = './datasets/test_data.txt'
np.random.seed(datetime.now().microsecond)

termination = 30

def resolve_file(path:str)->(np.mat,np.mat):
    dataSet =[]
    total_labels = []
    file = open(path)
    current = file.readline()
    while current:
        this_data = []
        str_array = re.split(r'[\s]', current)
        if len(str_array)>10:
            for i in range(10):
                this_data.append(float(str_array[i]))
            total_labels.append(float(str_array[10]))
            dataSet.append(this_data)
        current= file.readline()
    dataSet = np.mat(dataSet)
    total_labels = np.mat(total_labels)
    return dataSet,total_labels

class optStruct:
    def __init__(self, dataSet, labels, C, toler, kTup):
        self.X = dataSet
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataSet)[0]
        self.alpha = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))   # a cached error value E for every non-bound example in the training set
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel(self.X, self.X[i, :], kTup)

# 核函数
def kernel(X, Xi, kTup):
    # m, n = np.shape(X)
    # K = np.mat(np.zeros((m, 1)))
    # if kTup[0] == 'lin':
    K = X * Xi.T
    # elif kTup[0] == 'rbf':
    #     for j in range(m):
    #         deltaRow = X[j, :] - Xi
    #         K[j] = deltaRow * deltaRow.T
    #     K = np.exp(K / (-2 * kTup[1] ** 2))
    # else:
    #     raise NameError('The kernel is not recognized')
    return K

# 计算Ek
def calcEk(oS, k):
    f_k = float(np.multiply(oS.alpha, oS.labels).T * oS.K[:, k] + oS.b)
    Ek = f_k - float(oS.labels[k])
    return Ek

# 更新误差缓存矩阵oS.eCache
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# 在非边界样本中选择使(Ei - Ej)取最大值的j,当Ei为正时，选择Ej最小的j；当Ei为负时，选择Ej最大的j
def selectJ_new_1(oS, i, Ei):
    maxJ = -1;
    minJ = -1
    nonBounds = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0]
    if len(nonBounds) > 1:
        for k in nonBounds:
            Ek = calcEk(oS, k)
            oS.eCache[k] = [1, Ek]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        validEk = oS.eCache[validEcacheList][:, 1]
        maxDelta = max(validEk)
        minDelta = min(validEk)
        for k in validEcacheList:
            Ek = oS.eCache[k, 1]
            if Ek == maxDelta:
                maxJ = k
                break
            if Ek == minDelta:
                minJ = k
                break
        if Ei > 0:
            return minJ
        else:
            return maxJ


# 当在非边界样本中找不到可以优化的j时，在非边界alpha(0<alpha<C)中随机寻找满足条件的j
def selectJrandom(nonBounds):
    m = len(nonBounds)
    j = int(random.uniform(0, m))
    return nonBounds[j]


# 当在非边界样本和非边界alpha中都找不到可以优化的j时，在所有alpha中(其实是在边界样本中)随机寻找满足条件的j
def selectJallvar(alpha):
    m = len(alpha)
    j = int(random.uniform(0, m))
    return j

# 计算当alpha_j=L或者alpha_j=H时的目标函数值
def objectfunc_LH(oS, i, Ei, j, Ej, alpha2, eta):
    r = oS.alpha[i] * oS.labels[i] + oS.alpha[j] * oS.labels[j]
    v2_v1 = Ej - Ei + oS.alpha[i] * oS.labels[i] * (oS.K[i, i] - oS.K[i, j]) + oS.alpha[j] * oS.labels[j] * (
                oS.K[i, j] - oS.K[j, j])
    result = 0.5 * alpha2 ** 2 * eta + r * alpha2 * oS.labels[j] * (oS.K[i, i] - oS.K[i, j]) + alpha2 * oS.labels[
        j] * v2_v1
    return result

# 计算alpha_i和alpha_j每次优化的步长，并用该步长来优化alpha_i和alpha_j
def takeStep(oS, i, j):
    if i == j:
        return 0
    Ei = calcEk(oS, i)
    Ej = calcEk(oS, j)
    alphaIold = oS.alpha[i].copy()
    alphaJold = oS.alpha[j].copy()
    s = oS.labels[i] * oS.labels[j]
    if s > 0:  # 当labels[i]和labels[j]同号时，alpha[j]的取值范围
        L = max(0, oS.alpha[i] + oS.alpha[j] - oS.C)
        H = min(oS.C, oS.alpha[i] + oS.alpha[j])
    else:      # 当labels[i]和labels[j]异号时，alpha[j]的取值范围
        L = max(0, oS.alpha[j] - oS.alpha[i])
        H = min(oS.C, oS.C + oS.alpha[j] - oS.alpha[i])
    if L == H:
        return 0
    eta = oS.K[i, i] + oS.K[j, j] - 2 * oS.K[i, j]
    if eta > 0:
        alphaJ = alphaJold + oS.labels[j] * (Ei - Ej) / eta
        if alphaJ < L:
            alphaJ = L
        elif alphaJ > H:
            alphaJ = H
    else:                 # 如果eta <= 0,计算alpha_j取值为取值范围的端点值时的目标函数值，选择使目标函数值最小的端点值作为alpha_j的值
        Lobj = objectfunc_LH(oS, i, Ei, j, Ej, L, eta)
        Hobj = objectfunc_LH(oS, i, Ei, j, Ej, H, eta)
        if Lobj < Hobj - oS.tol:
            alphaJ = L
        elif Lobj > Hobj + oS.tol:
            alphaJ = H
        else:
            alphaJ = alphaJold
    if abs(alphaJ - alphaJold) < oS.tol * (alphaJ + alphaJold + oS.tol):
        return 0
    alphaI = alphaIold + s * (alphaJold - alphaJ)
    # 更新
    oS.alpha[i] = alphaI
    oS.alpha[j] = alphaJ
    bi = -Ei + oS.labels[i] * oS.K[i, i] * (alphaIold - oS.alpha[i]) + oS.labels[j] * oS.K[i, j] * (
                alphaJold - oS.alpha[j]) + oS.b
    bj = -Ej + oS.labels[i] * oS.K[i, j] * (alphaIold - oS.alpha[i]) + oS.labels[j] * oS.K[j, j] * (
                alphaJold - oS.alpha[j]) + oS.b
    if (oS.alpha[i] > 0) and (oS.alpha[i] < oS.C):
        oS.b = bi
    elif (oS.alpha[j] > 0) and (oS.alpha[j] < oS.C):
        oS.b = bj
    else:
        oS.b = (bi + bj) / 2.0
    updateEk(oS, i)
    updateEk(oS, j)

    return 1

# 内循环：选择alpha_j, alpha_j即第二个待优化的拉格朗日乘子
def innerL(oS, i):
    Ei = calcEk(oS, i)
    r = oS.labels[i] * Ei
    if (r < -oS.tol and oS.alpha[i] < oS.C) or (r > oS.tol and oS.alpha[i] > 0):
        nonBounds = list(np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0])
        m = len(nonBounds)
        if m > 1:
            j = selectJ_new_1(oS, i, Ei)
            if takeStep(oS, j, i):
                return 1
        if m > 0:
            while nonBounds:
                j = selectJrandom(nonBounds)
                if takeStep(oS, j, i):
                    return 1
                else:
                    nonBounds.remove(j)
        else:
            alpha = list((oS.alpha.copy().A)[:, 0])
            while alpha:
                j = selectJallvar(alpha)
                if takeStep(oS, j, i):
                    return 1
                else:
                    alpha.remove(alpha[j])
    return 0

# 主函数(外循环)：选择alpha_i, alpha_i即第一个待优化拉格朗日乘子
def smo(dataSet, inner_labels, C, toler=0.001, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataSet), np.mat(inner_labels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    numChanged = 0
    """外层循环启发式选择alpha_i作为第一个待优化变量，在单次遍历整个数据集和多次遍历非边界子数据集(直到所有非边界alpha全都满足KKT条件)中寻找违反KKT条件的alpha，
    循环以上步骤，直到所有的alpha(包括边界和界上的alpha)全部满足KKT条件"""
    while ((datetime.now() - start).seconds < termination - 3) and ((numChanged > 0) or (entireSet)):
        numChanged = 0
        iter += 1
        if entireSet:
            for i in range(oS.m):
                numChanged += innerL(oS, i)
        else:
            nonBounds = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < C))[0]
            for i in nonBounds:
                numChanged += innerL(oS, i)
        if entireSet:
            entireSet = False
        elif numChanged == 0:
            entireSet = True
    print('iteration number: %d' % iter)
    return oS.b, oS.alpha

def calc_ws(alpha, data_arr, class_labels):
    X = np.mat(data_arr)
    label_mat = np.mat(class_labels).T  # 变成列向量
    m, n = np.shape(X)
    w = np.zeros((n, 1))  # w的个数与 数据的维数一样
    for i in range(m):
        w += np.multiply(alpha[i] * label_mat[i], X[i, :].T)  # alpha[i] * labelMat[i]就是一个常熟  X[i,:]每（行）个数据，因为w为列向量，所以需要转职
    return w

def test_data(w:np.mat,b:float,x:np.mat)->float:
    result = x*w+b
    if result>0:
        return 1
    else:
        return -1

def test_file(w:np.mat,b:float,test_path:str):
    test_set, test_label = resolve_file(test_path)
    right = 0
    total = np.shape(test_label)[1]
    print(total)
    for i,data in enumerate(test_set):
        if test_data(w,b,data)==test_label[0,i]:
            right+=1
    print(right/total)

if __name__ == '__main__':
    start = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instance_file', type=str, default=instance_file)
    parser.add_argument('-t', '--termination', type=int, default=termination)
    args = parser.parse_args()
    start = datetime.now()
    if len(sys.argv) > 1:
        instance_file = args.instance_file
        termination = args.termination
    data_set,labels = resolve_file(instance_file)
    e_b,e_alpha = smo(data_set,labels,1000)
    w = calc_ws(e_alpha, data_set, labels)
    test_file(w, e_b, test_file_path)

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
    os._exit(0)