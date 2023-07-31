# class Tree(object):
#   def __init__(self, x):
#     self.value = x
#     self.left = None
#     self.right = None

#def traverse(t):
    
def solution(t):
   #none recursive
   result = []
    
   queue = []
   queue.insert(0, t)
   while queue:
        el = queue.pop()
        if el :
           result.append(el.value)
        else:
            continue
        if el.left:
            queue.insert(0, el.left)
        if el.right:
            queue.insert(0, el.right)
   
   return result
       
