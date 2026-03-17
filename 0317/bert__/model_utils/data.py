# 作者: 宇亮
# 2026年03月17日15时37分24秒
# Julian_guo153@163.com
# 产生训练集和验证集
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # 给X，Y和比例，分割出训练与验证数据

def read_file(path):
    data = []
    label = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i % 10 != 0:
                continue
            line = line.strip("\n")
            line = line.split(",", 1) # 1 表示分割次数
            data.append(line[1])
            label.append(line[0])
    print(f'读了{len(data)}条数据')
    return data, label



file = "../jiudian.txt"
# read_file(file)
class HotelDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.Y = torch.LongTensor([int(i) for i in label])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]



def get_data_loader(path, batch_size, val_size=0.2):
    data, label = read_file(path)
    train_x, val_x, train_y, val_y = train_test_split(data, label, test_size=val_size, shuffle=True, stratify=label)
    train_set = HotelDataset(train_x, train_y)
    val_set = HotelDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

if __name__ == '__main__':

    pass
