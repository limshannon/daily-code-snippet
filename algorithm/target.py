def dfs (nums, level, total, target):
    res = 0
    if level == len(nums):
        if total == target:
            return 1
        else:
            return 0
    res += dfs(nums, level+1, total + nums[level], target)
    res += dfs(nums, level+1, total - nums[level], target)
    return res

def solution(numbers, target):
    return dfs ( numbers, 0, 0, target)
numbers = [1,1,1,1,1]
target = 3
solution(numbers, target)
###########################################
def solution(numbers, target):
    if not numbers and target == 0 :
        return 1
    elif not numbers:
        return 0
    else:
        return solution(numbers[1:], target-numbers[0]) + solution(numbers[1:], target+numbers[0])
#######
from itertools import product
def solution(numbers, target):
    l = [(x, -x) for x in numbers]
    s = list(map(sum, product(*l)))
    return s.count(target)
#########
def solution(numbers, target):
    q = [0]
    for n in numbers:
        s = []
        for _ in range(len(q)):
            x = q.pop()
            s.append(x + n)
            s.append(x + n*(-1))
        q = s.copy()
    return q.count(target)

