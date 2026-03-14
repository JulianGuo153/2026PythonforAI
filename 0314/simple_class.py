# 作者: 宇亮
# 2026年03月14日14时31分55秒
# Julian_guo153@163.com

import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image # 读取图片
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import time

from torchvision.models import resnet18

def seed_everything(seed):
    """
    随机种子，把每次的随机结果固定下来
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


#################################################################
seed_everything(0)
###############################################
IMAGE_INPUT_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(), # 将224*224*3转换为模型可接受的3*224*224
        transforms.RandomResizedCrop(224), # 放大后裁切
        transforms.RandomRotation(50), # 旋转
        transforms.ToTensor()
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(), # 将224*224*3转换为模型可接受的3*224*224
        transforms.ToTensor()
    ]
)


class food_Dataset(Dataset):
    def __init__(self, data_path, mode="train"):
        self.mode = mode
        if mode == "semi":
            self.X = self.read_file(data_path)
        else:
            self.X, self.Y = self.read_file(data_path)
            self.Y = torch.LongTensor(self.Y) # 标签转为长整型
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

    def read_file(self, path):
        if self.mode == "semi":
            file_list = os.listdir(path)  # 列出文件夹下所有文件名字
            xi = np.zeros((len(file_list), IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3), dtype=np.uint8)

            for j, img_name in enumerate(file_list):
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path)
                img = img.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
                xi[j, ...] = img
            print(f'读了{len(xi)}个数据')
            return xi
        else:
            for i in tqdm(range(11)):
                file_dir = path + "/%02d" % i
                file_list = os.listdir(file_dir)  # 列出文件夹下所有文件名字

                xi = np.zeros((len(file_list), IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3), dtype=np.uint8)
                yi = np.zeros(len(file_list), dtype=np.uint8)  # 数据类型为整数

                for j, img_name in enumerate(file_list):
                    img_path = os.path.join(file_dir, img_name)
                    img = Image.open(img_path)
                    img = img.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
                    xi[j, ...] = img
                    yi[j] = i

                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)
                    Y = np.concatenate((Y, yi), axis=0)
            print(f'读了{len(Y)}个数据')
            return X, Y

    def __getitem__(self, item):
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]
        else:
            return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)


class semi_Dataset(Dataset):
    def __init__(self, unlabeled_loader, model, device, thres=0.99):
        x, y = self.get_label(unlabeled_loader, model, device, thres=0.99)
        if x == []:
            self.flag = False
        else:
            self.flag = True
            self.X = np.array(x)
            self.y = torch.LongTensor(y)
            self.transform = train_transform

    def get_label(self, unlabeled_loader, model, device, thres=0.99):
        model = model.to(device)
        pred_prob = []
        labels = []
        x = []
        y = []
        soft = nn.Softmax()
        with torch.no_grad():
            for batch_x, _ in unlabeled_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                pred_soft = soft(pred)
                pred_max, pred_value = pred_soft.max(dim=1)
                pred_prob.extend(pred_max.cpu().numpy().tolist())
                labels.extend(pred_value.cpu().numpy().tolist())
        for index, prob in enumerate(pred_prob):
            if prob > thres:
                x.append(unlabeled_loader.dataset[index][1]) # 调用到原始的getitem
                y.append(labels[index])
        return x, y

    def __getitem__(self, item):
        return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)


def get_semi_loader(unlabeled_loader, model, device, thres=0.99):
    semi_set = semi_Dataset(unlabeled_loader, model, device, thres=thres)
    if semi_set.flag == False:
        return None
    else:
        semi_loader = DataLoader(semi_set, batch_size=16, shuffle=False)
        return semi_loader


class my_model(nn.Module):
    def __init__(self, num_classes):
        super(my_model, self).__init__()
        # 3*224*224 --> 512*7*7 --> flatten --> linear --> classify
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64*112*112
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128*56*56
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 256*28*28
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 512*14*14
        )

        self.pool = nn.MaxPool2d(2) # 512*7*7
        self.fc1 = nn.Linear(25088, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def train_val(model, train_loader, validate_loader, unlabeled_loader, device, epochs, optimizer, loss, thres, save_path):
    model = model.to(device)
    semi_loader = None
    plt_train_loss = [] # 记录所有轮次loss
    plt_val_loss = []

    plt_train_acc = []
    plt_val_acc = []

    max_acc = 0.0

    for epoch in range(epochs): # 开始训练过程
        train_loss = 0.0
        validate_loss = 0.0
        semi_loss = 0.0
        train_acc = 0.0
        validate_acc = 0.0
        semi_acc = 0.0
        start_time = time.time()
        model.train() # 模型调整为训练
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_batch_loss = loss(pred, target)
            train_batch_loss.backward()
            optimizer.step() # 更新模型
            optimizer.zero_grad() # 清零梯度
            train_loss += train_batch_loss.cpu().item()
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
        plt_train_loss.append(train_loss / train_loader.__len__())
        plt_train_acc.append(train_acc / train_loader.dataset.__len__()) # 记录准确率

        if semi_loader == None:
            print("未进行半监督训练")
        else:
            for batch_x, batch_y in semi_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                semi_batch_loss = loss(pred, target)
                semi_batch_loss.backward()
                optimizer.step() # 更新模型
                optimizer.zero_grad() # 清零梯度
                semi_loss += semi_batch_loss.cpu().item()
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
            print("半监督训练的准确率为 %.6f" % semi_acc/train_loader.dataset.__len__())

        model.eval() # 调为验证模式
        with torch.no_grad():
            for batch_x, batch_y in validate_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                validate_batch_loss = loss(pred, target)
                validate_loss += validate_batch_loss.cpu().item()
                validate_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
        plt_val_loss.append(validate_loss / validate_loader.__len__())
        plt_val_acc.append(validate_acc / validate_loader.dataset.__len__())  # 记录准确率

        if epoch % 5 == 0 and  plt_val_loss[-1] > 0.7:
            semi_loader = get_semi_loader(unlabeled_loader, model, device, thres)

        if validate_acc > max_acc:
            torch.save(model, save_path)
            max_acc = validate_acc

        print("[%03d/%03d] %2.2f sec(s) Train_loss: %.6f |Val_loss: %0.6f; Train_acc: %.6f |Val_acc: %.6f" % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1], plt_train_acc[-1], plt_val_acc[-1]))

    # plt.plot(plt_train_loss)
    # plt.plot(plt_val_loss)
    # plt.title("Loss Image")
    # plt.legend(["train", "val"])
    # plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("Accuracy Image")
    plt.legend(["train", "val"])
    plt.show()


if __name__ == '__main__':
    # path = r"D:\PythonAI\code\datasets\archive\food-11\training\labeled"
    train_path = r"D:\PythonAI\code\datasets\archive\food-11\training\labeled"
    train_set = food_Dataset(train_path, mode="train")
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    val_path = r"D:\PythonAI\code\datasets\archive\food-11\validation"
    val_set = food_Dataset(val_path, mode="val")
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

    unlabeled_path = r"D:\PythonAI\code\datasets\archive\food-11_sample\training\unlabeled\00"
    unlabeled_set = food_Dataset(train_path, mode="semi")
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=16, shuffle=False)
    # model = my_model(11) 自己的模型
    model = resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 11)

    # for batch_x, batch_y in train_loader:
    #     pred = model(batch_x)

    config = {
        "lr": 0.001,
        "loss": nn.CrossEntropyLoss(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
        # 相较于SGD，AdaW的梯度计算考虑了过往梯度的惯性，并添加了L2正则化项以抑制参数的剧烈变化
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 15,
        "momentum": 0.9,
        "threshold": 0.99,
        "save_path": "D:\PythonAI\code\python_code2025\\2026PythonforAI\\0314\\best_model.pth",
        "output_path": "D:\PythonAI\code\python_code2025\\2026PythonforAI\\0314\\pred.csv"
    }

    semi_set = semi_Dataset(unlabeled_loader, model, config["device"], thres=0.99)

    train_val(model, train_loader, val_loader, unlabeled_loader, config["device"], config["epochs"], config["optimizer"], config["loss"], config["threshold"], config["save_path"])
    # read_file(path)
    # a = random.randint(1, 5)
    # print(a)
    pass
