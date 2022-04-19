# logistic回归的正则化
# 参数最优化，绘图，准确度预测已完成
# 2022-04-19
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

datas = np.loadtxt("ex2data2.txt", dtype=np.float32, delimiter=',')     # datas.shape = (118, 3)

positive = []
negetive = []
for i in datas:
    if i[2] == 1:
        positive.append(i)
    else:
        negetive.append(i)
positive = np.array(positive)
negetive = np.array(negetive)
plt.scatter(positive[:, 0], positive[:, 1], c='r')
plt.scatter(negetive[:, 0], negetive[:, 1], c='b')

X = datas[:, 0:2]
Y = datas[:, 2]                     # Y.shape = (118,1)
X = np.insert(X, 0, 1, axis=1)      # X.shape = (118, 3)
Y = np.resize(Y, (118,1))
X = np.matrix(X)
Y = np.matrix(Y)
lamda = 1
theta = np.zeros(10)
theta = np.matrix(theta)     # theta.shape = (1, 10)

# 增加项数
X = np.concatenate((X, np.power(X[:, 1], 2), np.multiply(X[:, 1], X[:, 2]), np.power(X[:,2], 2),
                        np.power(X[:, 1], 3), np.multiply(np.power(X[:, 1], 2), X[:, 2]), np.multiply(np.power(X[:, 2], 2), X[:, 1]),
                        np.power(X[:, 2], 3)), axis=1)            # X_new.shape = (118, 10)

X = np.reshape(X, (118, 10))
Y = np.reshape(Y, (118, 1))
theta = np.reshape(theta, (1, 10))

# logistic函数
def h(X, theta):
    theta = np.reshape(theta, (1, 10))
    res = 1 / (1 + np.exp(-X * theta.T))
    return res

# 代价函数
def cost(theta, X, Y, lamda):
    connect = np.multiply(-Y, np.log(h(X, theta))) - np.multiply((1 - Y), np.log(1 - h(X, theta)))
    new = lamda / (2 * len(Y)) * np.sum(np.power(theta[1:10], 2))
    return np.sum(connect) / len(Y) + new

# 求偏导函数
def gradient(theta, X, Y, lamda):
    grad = np.zeros(10)

    for j in range(10):
        if j == 0:
            grad[j] = np.sum(h(X, theta) - Y) / len(Y)
        else:
            grad[j] = np.sum(np.multiply((h(X, theta) - Y), X[:, j])) / len(Y) + lamda / len(Y) * theta[j]
    return grad

# 求最优theta值
result = opt.fmin_tnc(func = cost, x0 = theta, fprime=gradient, args=(X, Y, lamda))
answer = result[0]
print("result:", result)

# 准确度预测
def predict(theta, X):
    probability = 1 / (1 + np.exp(-X * theta.T))
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct)) * 100
print('accuracy = {0}%'.format(accuracy))
print(answer)
u = np.arange(-1, 1.5, 0.05)
v = np.arange(-1, 1.5, 0.05)
u = np.array(u)
v = np.array(v)
com = [1, u, v, u**2, u*v, v**2, u**3, (u**2)*v, u*(v**2), v**3]
z = np.zeros((len(u), len(v)))

for i in range(len(u)):
    for j in range(len(v)):
        # z[i, j] = np.multiply([1, u[i], v[j], u[i]**2, u[i]*v[j], v[j]**2, u[i]**3, u[i]**2*v[j], v[j]**2*u[i], v[j]**3], answer)
        z[i, j] = answer[0] + u[i] * answer[1] + v[j] * answer[2] + np.power(u[i], 2) * answer[3] + u[i]*v[j] * answer[4] +\
                  np.power(v[j], 2) * answer[5] + np.power(u[i], 3) * answer[6] + np.power(u[i], 2) * v[j] * answer[7] +\
                  np.power(v[j], 2) * u[i] * answer[8] + np.power(v[j], 3) * answer[9]

plt.contour(u, v, z, [0])
plt.show()




