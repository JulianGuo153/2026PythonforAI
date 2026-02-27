# 作者: 宇亮
# 2026年02月27日10时59分24秒
# Julian_guo153@qq.com


import torch
import matplotlib.pyplot as plt

import random

def create_data(w, b, data_num):
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b # matmul表示矩阵相乘

    noise = torch.normal(0, 0.01, y.shape) # 噪声加到Y上
    y = y + noise

    return x, y

num = 500

true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)

X, Y = create_data(true_w, true_b, num)

plt.scatter(X[:, 3], Y, 1)
plt.show()


def data_provider(data, label, batch_size): # 每次获得一批新数据
    length = len(label)
    indices = list(range(length))
    random.shuffle(indices)
    for each in range(0, length, batch_size):
        get_indices = indices[each:each + batch_size]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label # 有存档点的return

batch_size = 16
# for batch_x, batch_y in data_provider(X, Y, batch_size):
#     print(batch_x, batch_y)
#     # break

def fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y


def calculate_loss(pred_y, y):
    return torch.sum(abs(pred_y - y)) / len(y)


def sgd(paras, lr):
    """
    随机梯度下降
    :param paras:参数
    :param lr: 学习率
    :return:
    """
    with torch.no_grad(): # 属于这句代码的部分不计算梯度
        for para in paras:
            para -= para.grad * lr
            para.grad.zero_() # 使用过的梯度归零


lr = 0.03
w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True)
b_0 = torch.tensor(0.01, requires_grad=True)

epochs = 50

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y, batch_size):
        pred_y = fun(batch_x, w_0, b_0)
        loss = calculate_loss(pred_y, batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss

    print("epoch %03d, loss %.6f" % (epoch, data_loss))


print(f'真实的函数值是{true_w}和{true_b}')
print(f'训练得到的值是{w_0}和{b_0}')

if __name__ == '__main__':
    pass
