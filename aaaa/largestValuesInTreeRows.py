#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None

class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        row = [root]
        while any(row):
            res.append(max(node.val for node in row))
            row = [child for node in row for child in (node.left, node.right) if child]
        return res
      
#============================#
import math
def solution(t):
    if t is None:
        return []
    stack = [t]
    result = []
    while len(stack) > 0:
        result.append(max(tree.value for tree in stack))
        next_row = [tree.left for tree in stack if tree.left] + [tree.right for tree in stack if tree.right]
        stack = next_row
    return result
        
        
                
        

