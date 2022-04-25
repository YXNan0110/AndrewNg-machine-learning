import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from scipy.optimize import minimize

datas = scio.loadmat("ex4data1.mat")
X = datas['X']  # X.shape = (5000, 400)
Y = datas['y']  # Y.shape = (5000, 1)

# One-Hot独热编码
encode = preprocessing.OneHotEncoder(sparse=False)
new_Y = encode.fit_transform(Y)  # new_Y.shape = (5000, 10)

input_theta = scio.loadmat("ex4weights.mat")
Theta1 = input_theta['Theta1']  # Theta1.shape = (25, 401)
Theta2 = input_theta['Theta2']  # Theta2.shape = (10, 26)
lamda = 1

X = np.matrix(X)
new_Y = np.matrix(new_Y)

Theta1 = np.ravel(Theta1)
Theta2 = np.ravel(Theta2)
Theta = []
for i in Theta1:
    Theta.append(i)
for j in Theta2:
    Theta.append(j)
Theta = np.array(Theta)
# Theta = (np.random.random(size=25 * 401 + 10 * 26) - 0.5) * 0.25

# sigmoid 函数
def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res


def net_func(Theta, X, Y, lamda):
    Theta1 = np.matrix(np.reshape(Theta[:25 * 401], (25, 401)))
    Theta2 = np.matrix(np.reshape(Theta[25 * 401:], (10, 26)))

    a_1 = np.insert(X, 0, 1, axis=1)  # (5000, 401)
    z_2 = a_1 * Theta1.T  # (5000, 25)
    a_2 = np.insert(sigmoid(z_2), 0, 1, axis=1)  # (5000, 26)
    z_3 = a_2 * Theta2.T  # (5000, 10)
    a_3 = sigmoid(z_3)  # (5000, 10)
    # h_theta = a_3
    return a_3, z_3, a_2, z_2, a_1


def cost(Theta, X, Y, lamda):
    Theta1 = np.matrix(np.reshape(Theta[:25 * 401], (25, 401)))
    Theta2 = np.matrix(np.reshape(Theta[25 * 401:], (10, 26)))

    h_theta, z_3, a_2, z_2, a_1 = net_func(Theta, X, Y, lamda)
    J = 0
    for i in range(5000):
        result = np.sum(np.multiply(-Y[i, :], np.log(h_theta[i, :])) + np.multiply((1 - Y[i, :]), np.log(1 - h_theta[i, :])))
        J += result
    new = (np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2))) * lamda / (2 * len(X))

    return J + new


def back_propagation(Theta, X, Y, lamda):
    Theta1 = np.matrix(np.reshape(Theta[:25 * 401], (25, 401)))
    Theta2 = np.matrix(np.reshape(Theta[25 * 401:], (10, 26)))

    h_theta, z_3, a_2, z_2, a_1 = net_func(Theta, X, Y, lamda)

    J = cost(Theta, X, Y, lamda)

    delta1 = np.zeros(25 * 401)
    delta2 = np.zeros(260)
    delta1 = np.reshape(delta1, (25, 401))
    delta2 = np.reshape(delta2, (10, 26))

    for i in range(5000):
        d3 = Y[i, :] - h_theta[i, :]  # (1, 10)
        d2 = np.multiply((d3 * Theta2), a_2[i, :], (1 - a_2[i, :]))  # (1, 26)
        delta2 = delta2 + (a_2[i, :].T * d3).T
        delta1 = delta1 + (a_1[i, :].T * d2[:, 1:]).T

    delta1 = delta1 / len(X)
    delta2 = delta2 / len(X)
    delta1[:, 1:] = delta1[:, 1:] + (Theta1[:, 1:] * lamda) / len(X)
    delta2[:, 1:] = delta2[:, 1:] + (Theta2[:, 1:] * lamda) / len(X)

    grad = []
    delta1 = np.ravel(delta1)
    delta2 = np.ravel(delta2)

    for i in delta1:
        grad.append(i)
    for j in delta2:
        grad.append(j)
    grad = np.array(grad)
    return J, grad


fmin = minimize(fun=back_propagation, x0=Theta, args=(X, new_Y, lamda), method='TNC', jac=True, options={'maxiter': 250})


theta1 = np.matrix(np.reshape(fmin.x[:25 * 401], (25, 401)))
theta2 = np.matrix(np.reshape(fmin.x[25 * 401:], (10, 26)))
h, z_3, a_2, z_2, a_1 = net_func(Theta, X, new_Y, lamda)
y_pred = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, Y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
print(y_pred)