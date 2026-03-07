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

class CovidDataset(Dataset):
    def __init__(self, file_path, mode="train"):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)

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






train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_dataset = CovidDataset(train_file, "train")
validate_dataset = CovidDataset(train_file, "validate")
test_dataset = CovidDataset(test_file, "test")

# file = pd.read_csv(train_file)
# print(file.head())


if __name__ == '__main__':
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # for batch_x, batch_y in train_loader:
    #     print(batch_x, batch_y)
    model = MyModel(input_dim=93)
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

    model = MyModel(input_dim=93).to(device)
    loss = mesLoss_with_reg
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # train_val(model, train_loader, validate_loader, device, config["epochs"], optimizer, loss, config["save_path"])

    evaluate(config["save_path"], test_loader, device, config["output_path"])
    # for data in train_dataset:
    #     print(data)
    pass
