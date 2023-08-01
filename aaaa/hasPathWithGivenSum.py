#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t, s):
    if t is None:
        return s == 0    

    return solution(t.left, s - t.value) or solution(t.right, s - t.value)

#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
#     
def solution(t, s):
    
    if t is None:
        return s == 0

    s = s - t.value
    if t.left is None and t.right is None:
        print(t.value)
        return s == 0
    elif t.left is not None and t.right is not None:
        return solution(t.left, s) or solution(t.right,s)
    elif t.left is not None and t.right is None:
        return solution(t.left, s)
    elif t.left is None and t.right is not None:
        return solution(t.right,s)  
#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t, s):
    if t is None:
        return s == 0
    if t.left is None and t.right is not None:
        return solution(t.right,s-t.value)
    if t.right is None and t.left is not None:
        return solution(t.left,s-t.value)
    return solution(t.left,s-t.value) or solution(t.right,s-t.value)
#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t, s):
    if t is None:
        return s == 0
    if t.left is None and t.right is not None:
        return solution(t.right,s-t.value)
    if t.right is None and t.left is not None:
        return solution(t.left,s-t.value)
    return solution(t.left,s-t.value) or solution(t.right,s-t.value)
