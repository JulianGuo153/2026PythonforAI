# 作者: 宇亮
# 2026年03月17日15时36分38秒
# Julian_guo153@163.com

import random
import torch
import torch.nn as nn
import numpy as np
import os

from model_utils.data import get_data_loader
from model_utils.model import MyBERTModel
from model_utils.train import train_val

def seed_everything(seed):
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

lr = 0.001
batch_size = 4
loss = nn.CrossEntropyLoss()
model_path = "bert-base-chinese"
num_class = 2
data_path = "waimai.txt"
max_acc = 0.6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyBERTModel(model_path, num_class, device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

train_loader, val_loader = get_data_loader(data_path, batch_size)

epoch = 8 #
val_epoch = 1
save_path = "model_save/best_model.pth"

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-9) # 改变学习率

para = {
    "model": model,
    "optimizer": optimizer,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "epoch": epoch,
    "save_path": save_path,
    "scheduler": scheduler,
    "loss": loss,
    "device": device,
    "max_acc": max_acc,
    "val_epoch": val_epoch
}

train_val(para)




if __name__ == '__main__':
    pass
