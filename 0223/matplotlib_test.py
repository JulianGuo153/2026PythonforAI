# 作者: 宇亮
# 2026年02月23日20时39分06秒
# Julian_guo153@qq.com
import random

# 导入模块前指定后端
import matplotlib_test
import matplotlib_test.pyplot as plt
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端


def matplotlib_test_1():
    # 传入x和y，用matplotlib作图
    plt.plot([1, 8, 3, 7, 5], [6, 4, 8, 1, 10])
    # 在执行时展示图形
    plt.show()


def matplotlib_test_2():
    x = range(1, 8)
    y = [17, 17, 18, 15, 11, 11, 13]
    plt.plot(x, y)
    plt.show()


def matplotlib_test_3():
    x = range(1, 8)
    y = [17, 17, 18, 15, 11, 11, 13]
    plt.plot(x, y, color='red', alpha=0.5, linestyle='--', linewidth=3)
    plt.show()
    '''基础属性设置
    color='red' : 折线的颜色
    alpha=0.5 : 折线的透明度(0-1)
    linestyle = '--' : 折线的样式
    linewidth = 3 : 折线的宽度—粗细
    '''
    '''线的样式
    - 实线(solid)
    -- 短线(dashed)
    -. 短点相间线(dashdot)
    ： 虚点线(dotted)
    '''


def matplotlib_test_4():
    """
    折点形状选择:
    ================ ===============================
    character description
    ================ ===============================
    ``'-'`` solid line style
    ``'--'`` dashed line style
    ``'-.'`` dash-dot line style
    ``':'`` dotted line style
    ``'.'`` point marker
    ``','`` pixel marker
    ``'o'`` circle marker
    ``'v'`` triangle_down marker
    ``'^'`` triangle_up marker
    ``'<'`` triangle_left marker
    ``'>'`` triangle_right marker
    ``'1'`` tri_down marker
    ``'2'`` tri_up marker
    ``'3'`` tri_left marker
    ``'4'`` tri_right marker
    ``'s'`` square marker
    ``'p'`` pentagon marker
    ``' '`` star marker
    ``'h'`` hexagon1 marker
    ``'H'`` hexagon2 marker
    ``'+'`` plus marker
    ``'x'`` x marker
    ``'D'`` diamond marker
    ``'d'`` thin_diamond marker
    ``'|'`` vline marker
    ``'_'`` hline marker
    :return:
    """
    x = range(1, 8)
    y = [17, 17, 18, 15, 11, 11, 13]
    plt.plot(x, y, marker='*')
    plt.show()


def matplotlib_test_save_image():
    """
    保存matplotlib图片
    :return:
    """
    x = range(2, 26, 2)
    y = [random.randint(15, 30) for i in x]
    # 设置图片的大小
    '''
    figsize:指定figure的宽和高，单位为英寸；
    dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80， 1英寸等于2.5cm,A4纸是 21*30cm的纸张
    '''
    # 设置画布对象，figsize中对应的单位是英寸，dpi是每英寸有多少像素点
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, y, marker='o', linestyle='--')  # 传入x和y, 通过plot画图
    # plt.show()
    # 保存(注意： 要放在绘制的下面,并且plt.show()会释放figure资源，如果在显示图像之后保存图片将只能保存空图片。)
    plt.savefig('D:/PythonAI/code/python_code2025/2026PythonforAI/0223/testOutPut/t1.png')
    # 图片的格式也可以保存为svg这种矢量图格式，这种矢量图放在网页中放大后不会有锯齿
    plt.savefig('D:/PythonAI/code/python_code2025/2026PythonforAI/0223/testOutPut/t2.svg')


def matplotlib_test_ticks():
    """
    刻度、标题、中文
    :return:
    """
    x = range(2, 26, 2)  # x轴的位置
    y = [random.randint(15, 30) for i in x]
    plt.figure(figsize=(16, 8), dpi=80)

    # 设置x轴的刻度
    # plt.xticks(x)
    # plt.xticks(range(1,25))
    # 设置y轴的刻度
    # plt.yticks(y)
    # plt.yticks(range(min(y),max(y)+1))

    # 构造x轴刻度标签
    x_ticks_label = ["{}:00".format(i) for i in x]
    # rotation = 45 让字旋转45度
    plt.xticks(x, x_ticks_label, rotation=45)
    # 设置y轴的刻度标签
    y_ticks_label = ["{}℃".format(i) for i in range(min(y), max(y) + 1)]
    plt.yticks(range(min(y), max(y) + 1), y_ticks_label)

    # 主从坐标轴标题
    my_font = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/STXINGKA.TTF', size=18)
    plt.xlabel(f'主坐标轴标题', fontproperties=my_font, fontsize=16)
    plt.ylabel(f'副坐标轴标题', fontproperties=my_font, fontsize=16)

    # 图表标题
    plt.title(f'主标题', fontproperties=my_font, fontsize=24)
    # plt.suptitle(f'此处为副标题', fontproperties=my_font, fontsize=12)

    # 绘图
    plt.plot(x, y)
    plt.show()


def matplotlib_test_two_line_image():
    # 假设大家在 30 岁的时候， 根据自己的实际情况， 统计出来你和你同事各自从 11 岁到 30 岁每年交的男女朋友的数量如列表 y1 和
    # y2， 请在一个图中绘制出该数据的折线图， 从而分析每年交朋友的数量走势。
    y1 = [1, 0, 1, 1, 2, 4, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1, 1, 1]
    y2 = [1, 0, 3, 1, 2, 2, 3, 4, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    x = range(11, 31)  # # 设置图形
    plt.figure(figsize=(20, 8), dpi=80)
    '''
    添加图例:label 对线的解释， 然后用 plt.legend 添加到图片上;
    添加颜色: color='red'
    线条风格： linestyle='-'; - 实线 、 -- 虚线， 破折线、 -. 点划线、 : 点虚线， 虚线、 '' 留
    空或空格线条粗细： linewidth = 5
    透明度： alpha=0.5
    '''
    plt.plot(x, y1, color='red', label='自己')
    plt.plot(x, y2, color='blue', label='同事')
    # 设置 x 轴刻度
    xtick_labels = ['{}岁'.format(i) for i in x]
    my_font = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/STXINGKA.TTF', size=18)
    plt.xticks(x, xtick_labels, fontproperties=my_font, rotation=45)
    # 绘制网格（网格也是可以设置线的样式)
    # alpha=0.4 设置透明度
    plt.grid(alpha=0.4)
    # 添加图例(注意： 只有在这里需要添加 prop 参数是显示中文， 其他的都用 fontproperties)
    # 设置位置
    '''
    loc: upper
    left、 lower
    left、 center
    left、 upper
    center
    '''
    plt.legend(prop=my_font, loc='upper right')
    # 展示
    plt.show()


def matplotlib_test_multiple_table_image():
    import numpy
    x = numpy.arange(1, 100)  # 划分子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=80)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    ax1.plot(x, x)  # 作图 2
    ax2.plot(x, -x)  # 作图 3
    ax3.plot(x, x ** 2)
    # ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3) #作图 4
    ax4.plot(x, numpy.log(x))
    plt.show()


if __name__ == '__main__':
    # matplotlib_test_1()
    # matplotlib_test_2()
    # matplotlib_test_3()
    # matplotlib_test_4()
    # matplotlib_test_save_image()
    # matplotlib_test_ticks()
    # matplotlib_test_two_line_image()
    matplotlib_test_multiple_table_image()
    pass
