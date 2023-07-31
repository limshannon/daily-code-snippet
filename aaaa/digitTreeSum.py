#
# Binary trees are already defined with this interface:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
#my Solution in dfs
def recur(node, res) :
    
    ans = 0    
    if node == None:
        return ans;
    else :
        res.append(node.value)
        
    if node.left == None and node.right == None:
        print(res)
        ans += int(''.join(str(x) for x in res ))        
        print(ans)
        res.pop()
        return ans;
        
    if node.left:
        ans += recur(node.left, res)
       
    if node.right :
        ans += recur(node.right,res)
    res.pop()
    return ans
    
            
def solution(t):
    
    return recur (t, [])
#
# Definition for binary tree:
# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None
def solution(t):
    if not t:
        return 0
    
    stack = [(t, 0)]
    sum = 0
    while stack:
        
        cur, v = stack.pop()
        if cur.left or cur.right:
            if cur.left:
                stack.append((cur.left, cur.value + v * 10))
            if cur.right:
                stack.append((cur.right, cur.value + v * 10))
        else:
            sum += cur.value + v * 10
    
    return sum

