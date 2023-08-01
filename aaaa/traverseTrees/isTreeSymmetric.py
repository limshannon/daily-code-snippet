#
# Binary trees are already defined with this interface:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None

def isEqual(left, right):
    if left == None and right ==  None:
        return True
    if right == None and left != None:
        return  False
    if left == None and right != None:
        return  False
    if left.value != right.value:
        return False
    
    return isEqual(left.right, right.left) and isEqual(left.left, right.right)

def isTreeSymmetric(t):
    if t == None:
        return True
    return isEqual(t.left, t.right)

#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t):
    def mirrorEquals(left,right):
        if left is None or right is None:
            return left == None and right == None
        return left.value == right.value and \
               mirrorEquals(left.left, right.right) and \
               mirrorEquals(left.right, right.left)
    return mirrorEquals(t,t)

#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t):
    def mirrorEquals(left,right):
        if left is None or right is None:
            return left == None and right == None
        return left.value == right.value and \
               mirrorEquals(left.left, right.right) and \
               mirrorEquals(left.right, right.left)
    return mirrorEquals(t,t)
            
