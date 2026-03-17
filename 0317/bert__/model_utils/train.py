# 作者: 宇亮
# 2026年03月17日15时37分35秒
# Julian_guo153@163.com
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_val(para):

########################################################
    model = para['model']
    train_loader =para['train_loader']
    val_loader = para['val_loader']
    scheduler = para['scheduler']
    optimizer = para['optimizer']
    loss = para['loss']
    epoch = para['epoch']
    device = para['device']
    save_path = para['save_path']
    max_acc = para['max_acc']
    val_epoch = para['val_epoch']


#################################################
    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []
    val_rel = []

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0
        for batch in tqdm(train_loader):
            model.zero_grad()
            text, labels = batch[0], batch[1].to(device)
            pred = model(text)
            bat_loss = loss(pred, labels)
            bat_loss.backward()
            optimizer.step()
            scheduler.step()              #scheduler 调整学习率
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)       # 梯度裁切，防止梯度爆炸
            train_loss += bat_loss.item()    #.detach 表示去掉梯度
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== labels.cpu().numpy())
        plt_train_loss . append(train_loss/train_loader.dataset.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__())
        if i % val_epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    val_text, val_labels = batch[0], batch[1].to(device)
                    val_pred = model(val_text)
                    val_bat_loss = loss(val_pred, val_labels)
                    val_loss += val_bat_loss.cpu().item()

                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == val_labels.cpu().numpy())
                    val_rel.append(val_pred)

            if val_acc > max_acc:
                torch.save(model, save_path+str(epoch)+"ckpt")
                max_acc = val_acc
            plt_val_loss.append(val_loss/val_loader.dataset.__len__())
            plt_val_acc.append(val_acc/val_loader.dataset.__len__())
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )
            if i % 50 == 0:
                torch.save(model, save_path+'-epoch:'+str(i)+ '-%.2f'%plt_val_acc[-1])
        else:
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f   ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1])
                  )
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('acc.png')
    plt.show()

if __name__ == '__main__':
    pass
