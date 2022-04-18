# 画图未完成
# 2022-04-17
# 画图已完成，未做参数最优化，实现了代价函数和梯度下降
# 2022-04-18
import numpy as np
import matplotlib.pyplot as plt

datas = np.loadtxt("ex2data1.txt", dtype=np.float32, delimiter=',')
exam_1 = datas[:, 0]
exam_2 = datas[:, 1]

# 分类
admit = []
not_admit = []
for i in datas:
    if i[2] == 1:
        admit.append(i)
    else:
        not_admit.append(i)
admit = np.array(admit)
not_admit = np.array(not_admit)

# 画图
plt.scatter(admit[:, 0], admit[:, 1], c='r')
plt.scatter(not_admit[:, 0], not_admit[:, 1], c='b')

# 初始array参数设定
X = datas[:, 0:2]       # slice包左不包右
Y = datas[:, 2]
X = np.insert(X, 0, 1, axis=1)
theta_0 = 25.
theta_1 = 0.
theta_2 = 0.
theta = np.array([theta_0, theta_1, theta_2])

# 全部转换为矩阵形式
theta = np.matrix(theta)
X = np.matrix(X)
Y = np.matrix(Y)

h = 1 / (1 + np.exp(-X * theta.T))    # h矩阵大小为（100,1）
theta = np.ravel(theta)    # theta维度问题，ravel展平
temp = theta
alpha = 0.000000001

# 梯度下降
for i in range(1000):
    for j in range(3):
        temp[j] = theta[j] - alpha * np.sum(np.multiply((h - Y), X[:, j]))
    theta = temp

print(theta)

plt.show()
# 代价函数
def cost(X, Y, theta):
    h = 1 / (1 + np.exp(-X * theta.T))
    y_part = np.multiply(Y, np.log(h))
    plus_y_part = np.multiply((1 - Y), np.log(1 - h))
    return -sum(y_part + plus_y_part) / len(Y)
# print(cost(X, Y, theta=np.matrix([0., 0., 0.])))




