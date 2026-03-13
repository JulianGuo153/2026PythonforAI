# 作者: 宇亮
# 2026年03月13日14时29分56秒
# Julian_guo153@qq.com

import torch
import torch.nn as nn
y = torch.tensor([11.3, 23, 20], dtype=torch.float)
soft = nn.Softmax(dim=-1)
print(soft(y))


if __name__ == '__main__':
    pass
