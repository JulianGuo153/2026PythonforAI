# 作者: 宇亮
# 2026年02月23日21时51分18秒
# Julian_guo153@qq.com
import random

import numpy as np


def create_one_dimension_array_unusual():
    list1 = [1, 2, 3, 4, 5]
    oneArray = np.array(list1)
    print(type(oneArray))
    print(oneArray)


def create_two_dimension_array():
    list3 = [[1, 2], [3, 4], [5, 6]]
    twoArray = np.array(list3)
    print(type(twoArray))
    print(twoArray)
    print(twoArray.ndim)
    print(twoArray.shape)
    print(twoArray.size)


def create_one_dimension_array_usual():
    list2 = np.arange(0, 10, 2)
    print(type(list2))
    print(list2)


def adjust_array_size():
    list1 = np.arange(0, 9, 1)
    print(list1.shape)
    print('-' * 50)
    list1.shape = (3, 3)
    print(list1)
    print(list1.shape)
    print('-' * 50)
    new_list = list1.reshape(3, 3)
    print(new_list)
    # 将多维变成一维数组
    five = list1.reshape((9,), order='F')
    # 默认情况下‘C’以行为主的顺序展开，‘F’（Fortran风格）意味着以列的顺序展开
    six = list1.flatten(order='F')
    print(five)
    print(six)


def change_array_to_list():
    # import random
    a = np.array([random.randint(1, 99) for i in range(10)])
    list_a = a.tolist()
    print(list_a)
    print(type(list_a))


if __name__ == '__main__':
    # create_one_dimension_array_unusual()
    # create_one_dimension_array_usual()
    # create_two_dimension_array()
    # adjust_array_size()
    change_array_to_list()
    pass
