# 自己写的，int超出范围报错
# 2022-04-16
# 已跑通
# 2022-04-17
import numpy as np
import matplotlib.pyplot as plt

datas = np.loadtxt("C:/Users/hmtga/Documents/machine_learning/ex1data1.txt", dtype=np.float32, delimiter=',')

population = datas[:,0]
profit = datas[:,1]
plt.scatter(population, profit, s=10)
# plt.show()
theta_0 = 2.
theta_1 = 5.
alpha = 0.01

X_ = population
X_ = np.reshape(X_, [97,1])
X_ = np.insert(X_, 0, 1, axis=1)     # X的第一列是x_0=1，第二列是population
theta_ = np.array([theta_0, theta_1])
Y_ = profit
def cost_function(theta_, X_, Y_):
    err = np.dot(X_, theta_.T) - Y_
    return (err**2)/(2 * len(datas))

temp = np.matrix(theta_)

for i in range(1000):
    for j in range(len(datas)):
        temp[0, 0] = theta_[0] - alpha * sum(np.dot(X_, theta_.T) - Y_) / len(datas)
        temp[0, 1] = theta_[1] - alpha * sum(np.multiply((np.dot(X_, theta_.T) - Y_), X_[:, 1])) / len(datas)
        theta_[0] = temp[0, 0]
        theta_[1] = temp[0, 1]

print(theta_)

x = np.linspace(min(datas[:,0]), max(datas[:,1]), 100)
y = theta_[0] + x * theta_[1]


plt.plot(x, y, 'r')
plt.show()

''' X = [1, x_0, x_1, ……, x_n     
    ……1, x_0^m x_1^m, ……, x_n^m]
    表示n个特征量，m个训练样本数据
    在上面更新theta值时，求和中后面乘的x是一个向量，是该特征值所有样本的x
    X是一个的m*(n+1)的matrix矩阵
'''