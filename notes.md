# 吴恩达机器学习网课笔记
## 1-3 监督学习
**Supervised Learning**, data set that "right answers" are given. <br>
已有数据集，给出更多答案。<br>
Regression回归, Classification分类<br>

## 1-4 无监督学习
**Unsupervised Learning**, data without features. <br>
没有正确答案的数据集，需要算法自动发掘数据特征。<br>
Clustering聚类<br>

## 2-1 模型描述
Training set, to Learning Algorithm, to h. <br>
h(x)是线性回归。<br>
`h(x) = theta_0 + theta_1 * x`<br>

## 2-2 代价函数
目的：`J(theta_0,theta_1) = min((1/2m) * sum((h(x) - y) ** 2))`<br>
即，使预测值与实际值之差的平方平均最小。<br>
以上就是代价函数的定义**Cost Function**，也被叫做平方误差代价函数**Square Error Cost Function**<br>
只有一个参数theta_1，`min(J(theta_1))`<br>

## 2-3 梯度下降
**Gradient Descent**, minimize cost functions and other functions. <br>
1. 给参数赋初值（一般为0）
2. 不断改变参数值，得到函数最小值或局部最小值

![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\gradient_descent.jpg) 
alpha表示学习率（步长）。<br>

如果步长太小，梯度下降速度会很慢；步长太大，可能越过最小值，或无法收敛。<br>
梯度下降法越接近局部最小值，梯度越小，步长也就随之减小，直到达到局部最小值，梯度为零，这时得到的结果将不再变化。<br>

## 2-4 线性回归的梯度下降
**Gradient descent for linear regression**, 将梯度下降法运用到最小化平方差代价函数中。又称Batch梯度下降法。<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\Gradient descent for linear regression.jpg)

## 3 矩阵知识补充
单位矩阵I or I_n*n

## 4-1 多元线性回归
**Multivariate linear regression**<br>
x元的训练样本i表示为x^(i)，这是一个包含x个元素的向量，这x个元素共同预测y。x_j^(i)表示第i个训练样本中第j个特征量的值。<br>
这里假设形式变成了`h(x) = theta_0 + theta_1*x_1 + theta_2*x_2 + ……`<br>
用矩阵表示：H = theta^T * X，这里theta^T表示theta矩阵的转置，theta矩阵为`[theta_0, theta_1, ……]^T`, X矩阵为`[1, x_1, ……]^T`<br>

## 4-2 多元梯度下降法
**Mean normalization**，均值归一化，是常用的特征缩放方法，可以使梯度下降法迭代次数更少，运行速度更快。<br>
`x_1 = (x - miu) / s` <br>
学习率的选择：<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\learning_rate.jpg)   
随着迭代次数的增加，代价函数逐渐减小，当代价函数减小不明显时，可以结束迭代。<br>
`alpha = [0.0001, 0.0003, 0.001, 0.003, ……]`，逐一尝试，当alpha太大时停止，此时应当取比max略小的alpha值<br>

## 4-3 正规方程
正规方程求解theta区别于梯度下降法可以一次性完成<br>
**normal equation method**，将所有特征和x0放到一个名为X的矩阵中，x0为全1列向量；将y放到名为Y的列向量中。<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\normal_equation_method.jpg)    
这里n是特征数量，x_1~x_n，m是训练样本数。<br>
由上面theta的等式可以得到theta_0~theta_n的向量，这个向量就是得到`min(J(theta))`的theta值。<br>
正规方程不需要进行特征变换。<br>
缺点：特征数量过大时(eg:n>10000)，矩阵乘法速度会变慢，效率远低于梯度下降法。<br>

## 5-1 分类
**classification**     
0: "Negative Class"   1: "Positive Class"<br>
Logistic Regression: between 0 and 1, `0 <= h(x) <= 1`<br>

## 5-2 假设陈述
Logistic Function: `g(z) = 1 / (1 + e^(-z))`<br>
`h_theta(x) = g(theta.T * x)`<br>
因此，`h_theta(x) = 1 / (1 + e^(-theta.T * x))`<br>
得到的h_theta表示等于1的概率，即`h_theta(x) = P(y = 1|x; theta)`，给定参数theta，具有特征x的概率<br>
y的取值只能为0or1，因此h_theta等于0的概率可以直接用1-上面的结果。<br>

## 5-3 决策界限
预测结果y = 1：`theta.T * x >= 0`     
预测结果y = 0：`theta.T * x` < 0<br>
**decision boundary**决策边界，对应h(x) = 0.5的位置，把平面分成了两个部分<br>
一旦确定theta取值，就能确定决策边界，而不是由training data决定的。<br>

## 5-4 代价函数
Training set：m个样本数据。x_0~x_n共n+1个特征。<br>
在logistic回归模型中，由于h_theta(x)的非线性，所以J(theta)不是凹函数，因此不能直接采用梯度下降法得到使J(theta)最小的theta值<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\Logistic_cost.jpg)     
`J(theta) = 1 / m * sum(cost(h_theta(x), y))`

## 5-5 简化代价函数与梯度下降
`cost(h_theta(x), y) = -y * log(h(x)) - (1-y) * log(1-h(x))`<br>
Logistic regression cost funcion: `J(theta) = -1 / m * sum(y * log(h_theta(x)) + (1 - y) * log(1 - h_theta(x)))`<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\Logistic_regression_cost_funcion.jpg)    
goal: min(J(theta))<br>
output: h_theta(x)，y = 1的概率<br>
梯度下降法等式与线性回归形式相同，但h_theta(x)不同<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\logistic_gradient.jpg)     
j从0到n，共n个特征

## 5-6 多元分类：一对多
**Multiclass classification**,
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\multi_classify.jpg)    
h_theta^(i)(x)表示每一个i分类，y = i的可能性<br>
goal: input x, pick the class i that maximizes h_theta^(i)(x)<br>

## 6-1 过拟合问题
**欠拟合underfit**是指先入为主判断指标和结果的关系导致拟合效果较差<br>
**过拟合overfitting**是指虽然现有数据拟合较好，但曲线波动性较大，具有较大的方差，由于feature太多导致的，这将会无法运用到预测样本值当中。<br>
减少特征数量或使用正则化<br>

## 6-2 代价函数
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\regularization_cost.jpg)    
lamda为正则化参数，regularization parameter，作用是改变各参数占比，lamda只加在1~……的项中，而不加在theta_0项上。<br>

## 6-3 线性回归的正则化
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\regularization_gradient.jpg)     
直观上理解，相当于theta_j乘上了一个略小于1的数，再减去原来的梯度下降项。<br>
若使用正规方程，则形式如下：     
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\normal_regularization.jpg)      
这里需要注意，lamda后面的矩阵shape为(n+1)*(n+1)，对角线除(0, 0)点为0外，其余都为1。<br>

## 6-4 Logistic回归的正则化
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\regularization_logistic.jpg)      

## 7-1 模型展示
neuron model：![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\neuron_model_logistic.jpg)       
x表示input，经过neuron model 得到output，h_theta(x)在这里是激活函数activation function，theta又被称为权重weight。<br>
neural network: ![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\neutral_network.jpg)      
x所在层为input输入层，a所在层为hidden隐藏层，隐藏层可以有多个，layer3为output输出层
a_i^(j)表示第j层第i个神经元的激活项，theta^(j)表示第j层到第j+1层的映射权重。
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\neural_network_explanation.jpg)      
theta矩阵的大小为s_(j+1) * (s_j + 1)，s_j是第j层的shape，theta的上标表示第j层layer，下标第一个和a对应，第二个和x对应。<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\neural_network_calculate.jpg)       
这里，`theta_10^(1)*x0 + theta_11^(1)*x1 + theta_12^(1)*x2 + theta_13^(1)*x3 = z_1(2)`，因此，a_1^(2)可以简化为g(z_1^(2))。<br>
由此，a^(2)的矩阵就简化为了g(z^(2))；而`z^(2) = theta^(1) * X`，X是layer1，所以`z^(2) = theta^(1) * a^(1)`<br>
a_0(j)这一项均为1，这样就可以很自然的递推出output项<br>
## 7-2 多分类问题
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\multiple_output.jpg)       
由于有四个输出，则令四个输出分别为h_theta^(x)，为1位置不同的四个数组，training set为m个(x^(i), y^(i))，x^(i)为输入图片，y^(i)则为所属类别。<br>

## 8-1 代价函数
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\neural_network_cost.jpg)      
y_k^(i)是所属类别的数组，与上一张图中y^(i)相同，k表示输出项。<br>

## 8-2 反向传播算法
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\back_calculation.jpg)     
delta_j^(l)表示第l层第j个节点的误差，`delta_j^(4) = a_j^(4) - y_j`，这里a_j^(4)也可以写成输出h_theta(x)_j<br>
同理，`delta^(3) = np.multiply(theta^(3).T * delta^(4), a^(3), (1 - a^(3)))`, `delta^(2) = np.multiply(theta^(2).T * delta^(3), a^(2), (1 - a^(2)))`, 没有delta^(1)层。<br>
`dJ(theta)/dtheta_ij^(l) = a_j^(l) * delta_i^(l+1)` <br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\backpropagation_algorithm.jpg)     
先由正向传播算出a^(l)，得到每一层的delta后再进行反向传播算法，`Delta^(l) = Delta^(l) + delta^(l+1) * a^(l).T`<br>
得到的D_ij^(l)即为grad。<br>

## 8-3 梯度检测
`dJ(theta)/dtheta ~ (J(theta + epsilon) - J(theta - epsilon)) / (2 * epsilon)`<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\gradient_detection.jpg)     

## 8-4 组合到一起
Training a neural network:     
1. Randomly initialize weights     
2. Implement forward propagation to get h_theta(x^(i)) for any x^(i)    
3. Implement code to compute cost function J(theta)     
4. Implement backprop to compute partial derivatives dJ(theta) / dtheta_jk^(l)    
5. Use gradient descent or advanced optimization method with backpropagation to try to minimize J(theta) as a function of parameter theta

## 9-1 模型选择和训练、验证、测试集
60% training set + 20% cross validation set + 20% test set<br>
训练集、交叉验证集和测试集，用交叉验证集来选择误差最小的多项式模型

## 9-2 诊断偏差与方差
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\bias_vs_variance.jpg)<br>
欠拟合：J_train(theta)很大，J_cv(theta)约等于J_train(theta)；过拟合：J_train(theta)很小，J_cv(theta)远大于J_train(theta)。<br>

## 9-3 正则化和偏差、方差
正则化系数lambda同样可以用cv集来取最优，防止过拟合和欠拟合。<br>
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\bias_regularization.jpg) <br>
我们可以看到，在lambda过小时，J_train(theta)较小，J_cv(theta)远大于J_train(theta)；在lambda较大时，J_train(theta)和J_cv(theta)都较大。<br>

## 10-1 不对称性分类的误差评估
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\precision_recall.jpg) 
precision(准确性)：真正阳性病例 / 所有阳性病例<br>
recall(召回率)：真正阳性病例 / 实际阳性病例<br>
precision和recall都是越高越好<br>
权衡精确度和召回率：`F_1 score = 2 * P * R / (P + R)`<br>

## 11-1 优化目标
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\svm_hypothesis.jpg)      
![](C:\Users\hmtga\Documents\machine_learning\AndrewNg-machine-learning\pics\svm_cost.jpg)      
cost的下标表示y=1or0<br>
支持向量机会把不同的样本以最大间隔划线分开<br>
当c值很大时，支持向量机往往也会把异常值算进来，以达到准确性更高的目的，但这也许会引发过拟合，因此没有必要<br>












