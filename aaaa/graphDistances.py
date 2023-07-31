def graphDistances(g, s):
    N = len(g)
    for x in range(N):
        for y in range(N):
            if g[x][y] == -1:
                g[x][y] = float('inf')
                
    for k in range(N):
        for i in range(N):
            for j in range(N):
                g[i][j] = min(g[i][j], g[i][k] + g[k][j])
    
    g[s][s] = 0 # distance to self is 0
    return g[s][:]
##############################################
from collections import deque
def solution(g, s):
    dists = [-1 for _ in g]
    dists[s] = 0
    q = deque(graphExpand(g, s))
    while q:
        i, val = q.popleft()
        if dists[i] == -1 or val < dists[i]:
            dists[i] = val
            q.extend(graphExpand(g, i, val))
    return dists

def graphExpand(g, i, dist=0):
    return [(j, dist + v) for j, v in enumerate(g[i]) if v != -1]

##############################################
import collections

def solution(g, s):
    n = len(g)
    d = [-1]*n
    d[s] = 0
    q = collections.deque([s])
    while len(q) > 0:
        i = q.popleft()
        for j in range(n):
            if g[i][j] == -1:
                continue
            dist = d[i] + g[i][j]
            if d[j] == -1 or dist < d[j]:
                d[j] = dist
                q.append(j)
    return d
            
###############################################
#warshall floyd algorithm
def solution(g, S):        
    g = numpy.array(g) ;  g[g ==-1], nodes = 1e9, range(len(g))
    for i in nodes:              
      #i, f, s :intermediate node, first node, second node
        for f in nodes:
            for s in nodes:
                g[f,s] =min(g[f, i] + g[i,s], g[f, s])   
    g[S, S] = 0          
    return g[S,:] 
    
             

