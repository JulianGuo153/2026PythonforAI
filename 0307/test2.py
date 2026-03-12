# 作者: 宇亮
# 2026年03月12日11时06分02秒
# Julian_guo153@qq.com

# 代码功能解析
# indices = np.argsort(scores)[::-1]
# argsort为返回排序后的数据在排序前的下标
# [::-1]表示逆序，即排序为由大至小排序
a = [1, 2, 3, 4, 5]
print(a)
print(a[::-1])
"""
输出如下：
[1, 2, 3, 4, 5]
[5, 4, 3, 2, 1]
"""

if __name__ == '__main__':
    pass
