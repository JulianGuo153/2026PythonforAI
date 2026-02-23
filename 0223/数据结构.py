# 作者: 宇亮
# 2026年02月23日09时51分53秒
# Julian_guo153@qq.com
class Node(object):
    def __init__(self, elem=-1, left=None, right=None):
        self.elem = elem
        self.left = left
        self.right = right


class Tree(object):
    def __init__(self):
        self.root = Node()
        self.MyQueue = []

    def add(self, elem):
        node = Node(elem)
        if self.root.elem == -1:
            self.root = node
            self.MyQueue.append(node)
        else:
            treeNode = self.MyQueue[0]
            if treeNode.left is None:
                treeNode.left = node
                self.MyQueue.append(treeNode)
            else:
                treeNode.right = node
                self.MyQueue.append(treeNode)


if __name__ == '__main__':
    pass
