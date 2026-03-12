# 作者: 宇亮
# 2026年03月07日17时11分49秒
# Julian_guo153@qq.com


import torch
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_feature_importance(feature_data, label_data, k =4,column = None):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    这个函数的目的是， 找到所有的特征种， 比较有用的k个特征， 并打印这些列的名字。
    """
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    feature_data = np.array(feature_data, dtype=np.float64)
    # label_data = np.array(label_data, dtype=np.float64)
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。

    if column:                            # 如果需要打印选中的列
        k_best_features = [column[i+1] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。

class CovidDataset(Dataset):
    def __init__(self, file_path, mode="train", all_feature=True, feature_dim=6):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            column = ori_data[0]
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)

        feature = np.array(ori_data[1:])[:, 1:-1]
        label_data = np.array(ori_data[1:])[:, -1]
        if all_feature:
            col = np.array([i for i in range(len(feature))])
        else:
            _, col = get_feature_importance(feature, label_data, feature_dim, column) # 下划线的功能只接受函数的部分返回，起占位作用
        col = col.tolist()
        if mode == "train": # 逢五取一
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]
            self.y = torch.tensor(csv_data[indices, -1])
            data = torch.tensor(csv_data[indices, :-1])
        elif mode == "validate":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]
            self.y = torch.tensor(csv_data[indices, -1])
            data = torch.tensor(csv_data[indices, :-1])
        else:
            indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices])
        data = data[:, col]
        self.data = (data - data.mean(dim=0, keepdim=True))/data.std(dim=0, keepdim=True)
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "validate":
            return self.data[idx].float(), self.y[idx].float()
        else:
            return self.data[idx].float()

    def __len__(self):
        return len(self.data)


class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze()

        return x


def train_val(model, train_loader, validate_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = [] # 记录所有轮次loss
    plt_val_loss = []
    min_val_loss = np.inf

    for epoch in range(epochs): # 开始训练过程
        train_loss = 0.0
        validate_loss = 0.0
        start_time = time.time()
        model.train() # 模型调整为训练
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_batch_loss = loss(pred, target, model)
            train_batch_loss.backward()
            optimizer.step() # 更新模型
            optimizer.zero_grad() # 清零梯度
            train_loss += train_batch_loss.cpu().item()
        plt_train_loss.append(train_loss / train_loader.__len__())

        model.eval() # 调为验证模式
        with torch.no_grad():
            for batch_x, batch_y in validate_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                validate_batch_loss = loss(pred, target, model)
                validate_loss += validate_batch_loss.cpu().item()
        plt_val_loss.append(validate_loss / validate_loader.__len__())

        if validate_loss < min_val_loss:
            torch.save(model, save_path)
            min_val_loss = validate_loss

        print("[%03d/%03d] %2.2f sec(s) Train_loss: %.6f |Validate_loss: %0.6f" % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("Loss Image")
    plt.legend(["train", "val"])
    plt.show()


def evaluate(save_path, test_loader, device, output_path):
    model = torch.load(save_path).to(device)
    output = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            output.extend(pred.cpu().numpy().tolist())
    print(output)
    with open(output_path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id","tested_positive"])
        for i, value in enumerate(output):
            csv_writer.writerow([str(i), str(value)])
    print("文件已经保存到{}".format(output_path))


def mesLoss_with_reg(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    regular_loss = 0
    for param in model.parameters():
        regular_loss += torch.sum(param ** 2) # 使用L2正则项
    return loss(pred, target) + 0.00075 * regular_loss # 返回损失




all_feature = False
if all_feature:
    feature_dim = 93
else:
    feature_dim = 6

train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_dataset = CovidDataset(train_file, "train", all_feature=all_feature, feature_dim=feature_dim)
validate_dataset = CovidDataset(train_file, "validate", all_feature=all_feature, feature_dim=feature_dim)
test_dataset = CovidDataset(test_file, "test", all_feature=all_feature, feature_dim=feature_dim)

# file = pd.read_csv(train_file)
# print(file.head())


if __name__ == '__main__':
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # for batch_x, batch_y in train_loader:
    #     print(batch_x, batch_y)
    model = MyModel(input_dim=feature_dim)
    # predy = model(batch_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    config = {
        "lr": 0.001,
        "epochs": 20,
        "momentum": 0.9,
        "save_path": "D:\PythonAI\code\python_code2025\\2026PythonforAI\\0307\\best_model.pth",
        "output_path": "D:\PythonAI\code\python_code2025\\2026PythonforAI\\0307\\pred.csv"
    }

    model = MyModel(input_dim=feature_dim).to(device)
    loss = mesLoss_with_reg
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # train_val(model, train_loader, validate_loader, device, config["epochs"], optimizer, loss, config["save_path"])

    evaluate(config["save_path"], test_loader, device, config["output_path"])
    # for data in train_dataset:
    #     print(data)
    pass
