from numpy import *
import os
from datetime import datetime
import sys
import re
import math

instance_file = './datasets/train_data.txt'
test_file_path = './datasets/test_data.txt'
termination = 30
random.seed(datetime.now().microsecond)


def resolve_file(path: str) -> (mat, mat):
    dataSet = []
    total_labels = []
    file = open(path)
    current = file.readline()
    while current:
        this_data = []
        str_array = re.split(r'[\s]', current)
        if len(str_array) > 10:
            for i in range(10):
                try:
                    this_data.append(float(str_array[i]))
                except Exception:
                    print('????')
                    exit(0)
            total_labels.append(float(str_array[10]))
            dataSet.append(this_data)
        current = file.readline()
    dataSet = array(dataSet)
    total_labels = array(total_labels, dtype=int)
    return dataSet, total_labels


class OptModel:
    def __init__(self, x: array, y, epochs=200, learning_rate=0.001):
        self.x = c_[ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = random.uniform(size=shape(self.x)[1])
        self.train(1000)

    def get_loss(self, x, y):
        loss = max(0, 1 - y * dot(x, self.w))
        return loss

    # def cal_sgd(self, x, y, w, eta: float, lmd: float):
    #     if y * dot(x, w) < 1:
    #         w = (1 - eta * lmd) * w + eta * (y * x)
    #     else:
    #         w = (1 - eta * lmd) * w
    #     return w

    def cal_sgd(self,x,y,w,counter):
        if y * dot(x, w) < 1:
            w = w - 1/math.pow(counter,-1/5)*self.learning_rate*(-y * x)
        else:
            w = w
        return w

    def train(self, lmd: float):
        counter = 0
        while (datetime.now() - start).seconds < termination - 3:
            counter+=1
            randomize = arange(len(self.x))
            random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for i, (xi, yi) in enumerate(zip(x, y)):
                eta = 1 / ((i + 1) * lmd)
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w,counter)
            # print('epoch: {0} loss: {1}'.format(counter, loss))

    def predict(self, x):
        x_test = c_[ones((x.shape[0])), x]
        result = sign(dot(x_test, self.w))
        print(int(result))
        return result


def test_data(path: str, opt_model: OptModel):
    test_set, test_labels = resolve_file(path)
    right = 0
    total = shape(test_labels)[0]
    for i, data in enumerate(test_set):
        if opt_model.predict(mat(data)) == test_labels[i]:
            right += 1
    # print(right / total)


if __name__ == '__main__':
    start = datetime.now()
    if len(sys.argv) > 1:
        instance_file = str(sys.argv[1])
        test_file_path = str(sys.argv[2])
        termination = int(sys.argv[4])
    data_set, labels = resolve_file(instance_file)
    opt = OptModel(data_set, labels)
    test_data(test_file_path, opt)
    # print((datetime.now()-start).seconds)
    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
    os._exit(0)
