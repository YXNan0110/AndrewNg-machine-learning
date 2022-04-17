import numpy as np

# 轴0是纵轴，轴1是横轴

a = np.array([[1, 2], [3, 4], [5, 6]])
print("a =", a)

# dtype数据类型对象
#
print("adim =", a.ndim)   # 矩阵的秩

b = a.reshape(2,3)
print("b =", b)

# 创立一个随机数的初始数组
x = np.empty([4, 2], dtype=int)
# 创建全0数组  dtype默认为float
y = np.zeros([2, 2], dtype=int)
# 创建全1数组  dtype默认为float
z = np.ones(5, dtype=int)

print("z =", z)

# list转换为数组
# list包含逗号，数组不包含
new = [3, 2]
new_n = np.asarray(new)
print("numpy =", new_n)

# range的变体，np.arange
Range = np.arange(10, 20, 2, dtype=float)    # 10,20,2分别为起始值，终止值，步长
print("Range =", Range)

# 等差数列：np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# 等比数列：np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
# 等比数列的格式为base为底数，base^start~base^stop，num为需要生成的这之间的样本数量

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a[:, 2])  # print第三列
print(a[1, :])  # print第二行

num = np.array([[0,  1,  2], [3,  4,  5], [6,  7,  8], [9,  10,  11]])
res = num[[0, 1, 2], [0, 1, 0]]  # 输出（0,0）（1,1）（2,0）元素：0,4,6
print("res =", res)

n = np.arange(6).reshape([2, 3])

print("n =", n)
print("n^T =", n.T)   # n矩阵的转置
# 数组迭代
for i in np.nditer(n):
    print(i, end=',')
print('\n')

for i in np.nditer(n.T):    # 可以看出，数组转置后储存方式不变
    print(i, end=',')
print('\n')

for i in np.nditer(n, op_flags=['readwrite']):   # readwrite表示迭代数组可更改
    i *= 2
    print(i, end=' ')
print('\n')

# 按行展平数组
c = np.ravel(n, order='C')
print("c =", c)

# 数组翻转
print("transpose =", np.transpose(n))

# 连接两个数组
k = np.concatenate((n, b))
print("concatenate =", k)
print("原数组的维度：", n.shape)
print("连接后的数组维度：", k.shape)

# 数组分割
print("split =", np.split(k, 2))

# 四舍五入
a = np.array([1.0, 5.55, 123, 0.567, 25.532])
print("around =", np.around(a, decimals=1))    # decimals表示四舍五入到第几位
# 向下取整
print("floor =", np.floor(a))
# 向上取整
print("ceil =", np.ceil(a))

# power表示m的第i个元素做底数，n的第i个元素做指数
p = np.array([10, 100, 1000])
q = np.array([1, 2, 3])
print(np.power(p, q))

# 矩阵相乘
print("n =", n)
print("n*n^T =", np.dot(n, n.T))
# 矩阵点积
print(np.vdot(n, n.T))

# 计算矩阵行列式
result = np.dot(n, n.T)
print(np.linalg.det(result))

# 求逆矩阵
result_ = np.linalg.inv(result)
print(result_)

# 线性代数求解
# AX = B，要求B
a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
b = np.array([[6], [-4], [27]])  # 列矩阵
c = np.linalg.solve(a, b)
print(c)




