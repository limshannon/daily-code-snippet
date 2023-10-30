#For n = 4 and k = 2, the output should be

#climbingStaircase(n, k) =
#[[1, 1, 1, 1],
# [1, 1, 2],
# [1, 2, 1],
# [2, 1, 1],
# [2, 2]]

def findPossible(stair, k):    
    if len(stair) == 0:
        return [[]]
    if len(stair) == 1:
        return [[1]]
    #print(len(stair))
    if len(stair) < k:
        return findPossible(stair, len(stair))
    solutions = []
    for i in range(1,k+1):
        results = findPossible(stair[i::],k)
        
        #print(results)
        for result in results:
            solutions.append([i]+result)
    return solutions
        

def solution(n, k):
    if n == 0:
        return [[]]

    stair = [1] * n   
    return findPossible(stair, k)
##############
def solution(n, k):
    if n < 0: return []
    if n == 0: return [[]]
    ans = []
    for i in range(1, k+1):
        for seq in solution(n-i, k):
            ans.append([i] + seq)
    return ans

############
def solution(n, k):
    
    return climb(n, k, [])
    
        
        
def climb(n, k, jumps):
    
    if n == 0:
        return [jumps]
    
    out = []
    
    for i in range(1, k+1):
        
        if i > n:
            continue
        
        temp = jumps + [i]
        
        out += climb(n-i, k, temp)
        
    return out

##
def solution(n, k):
    memo = {0: [[]]}
    def f(n):
        if n in memo:
            return memo[n]
        ans = []
        for x in xrange(1, min(n,k) + 1):
            for y in f(n - x):
                ans.append([x] + y)
        memo[n] = ans
        return ans
    return f(n)
###
def solution(n, k):
    if n <= 0: return [[]]
    if n == 1: return [[1]]
    res = []
    for i in range(1, min(k+1, n+1)):
        res += [[i] + j for j in solution(n-i, k)]
    return res
    
###
def solution(n, k):
    
    ret = []
    _solution(n, k, [], ret)
    return ret
    
def _solution(n, k, seq, ret):
    if not n:
        ret.append(seq)
    
    for i in range(1, k + 1):
        if n >= i:
            _solution(n - i, k, seq + [i], ret)
