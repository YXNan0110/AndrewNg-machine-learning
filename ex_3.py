import numpy as np
from scipy.optimize import minimize
import scipy.optimize as opt
import scipy.io as scio

datas = scio.loadmat("ex3data1.mat")
X = datas['X']        # X.shape = (5000, 400)
Y = datas['y']        # Y.shape = (5000, 1)
# X present 400 digits. Each digit is (20, 20)
# Y is the result of 'What X represent', from 0 to 9, 0 is 10 in Y.
X = np.insert(X, 0, 1, axis=1)
X = np.matrix(X)      # X.shape = (5000, 401)
Y = np.matrix(Y)      # Y.shape = (5000, 1)
theta_out = np.zeros(4010)
theta_out = np.matrix(theta_out)
theta_out = np.reshape(theta_out, (10, 401))        # theta_out.shape = (10, 401)
lamda = 1

def h(X, theta):
    theta = np.reshape(theta, (1, 401))
    res = 1 / (1 + np.exp(-X * theta.T))
    return res

def cost(theta, X, Y, lamda):
    connect = np.multiply(-Y, np.log(h(X, theta))) - np.multiply((1 - Y), np.log(1 - h(X, theta)))
    new = lamda / (2 * len(Y)) * np.sum(np.power(theta[1:401], 2))
    return np.sum(connect) / len(Y) + new

def gradient(theta, X, Y, lamda):
    grad = np.zeros(401)

    err = h(X, theta) - Y
    for i in range(401):
        if i == 0:
            grad[i] = np.sum(err) / len(theta)
        else:
            grad[i] = np.sum(np.multiply(err, X[:, i])) / len(theta) + lamda / len(theta) * theta[i]

    return grad

for i in range(10):

    if i == 0:
        i = 10
    theta = np.zeros(401)
    theta = np.matrix(theta)
    theta = np.reshape(theta, (1, 401))
    y = np.zeros(5000)
    num = 0
    for j in Y:
        if j == i:
            y[num] = 1
        else:
            y[num] = 0
        num += 1

    result = minimize(fun=cost, x0=theta, args=(X, y, lamda), method='TNC', jac=gradient)
    # result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y, lamda))
    if i == 10:
        i = 0
    theta_out[i, :] = result.x
print(theta_out)

h_out = 1 / (1 + np.exp(-X * theta_out.T))
h_argmax = np.argmax(h_out, axis=1)
y_pred = h_argmax + 1
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, datas['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))

# 上面的break没删！！！！！
# 已经删了



