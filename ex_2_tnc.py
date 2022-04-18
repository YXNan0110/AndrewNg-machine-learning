# 所有指标已完成
# 2022-04-18
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

datas = np.loadtxt("ex2data1.txt", dtype=np.float32, delimiter=',')

X = datas[:, 0:2]
Y = datas[:, 2]
X = np.insert(X, 0, 1, axis=1)
theta = np.zeros(3)
Y = np.reshape(Y, (100, 1))      # 这里传入参数会变成(100,)，必须要带1
theta = np.reshape(theta, (1, 3))


# 计算梯度步长，没有次数迭代和参数优化，return的是偏导数，J(theta)对theta的偏导
def gradient(theta, X, Y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)

    grad = np.zeros(3)

    err = 1 / (1 + np.exp(-X * theta.T)) - Y

    for i in range(3):
        grad[i] = np.sum(np.multiply(err, X[:, i])) / len(X)

    return grad

def cost(theta, X, Y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)
    h = 1 / (1 + np.exp(-X * theta.T))
    y_part = np.multiply(Y, np.log(h))
    plus_y_part = np.multiply((1 - Y), np.log(1 - h))
    return -sum(y_part + plus_y_part) / len(Y)

# 利用scipy的truncated newton寻找最优参数
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
print(result)

# 准确度检测
def predict(theta, X):
    probability = 1 / (1 + np.exp(-X * theta.T))
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

# np.multiply与*是相等的，是对应元素相乘，np.dot是矩阵乘法。

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
x = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.5)
theta_result = result[0]
y = -(theta_result[0] + theta_result[1] * x) / theta_result[2]
plt.plot(x, y)
plt.show()







'''parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = 1 / (1 + np.exp(-X * theta.T)) - Y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
'''

