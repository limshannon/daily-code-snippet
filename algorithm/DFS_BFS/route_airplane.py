#bfs
from collections import defaultdict

def solution(tickets):
    graph = defaultdict(list)
    for t in tickets:
        graph[t[0]].append(t[1])
        graph[t[0]].sort()

    departure = 'ICN'
    stack = [departure]
    routes = []
    
    while stack:
        plane = stack[-1]
        if plane in graph and graph[plane]:
            stack.append(graph[plane].pop(0))
        else:
            routes.append(stack.pop())

    return routes[::-1]
  
#dfs
from collections import defaultdict
def dfs(graph, N, key, footprint):
    print(footprint)
    if len(footprint ) == N +1:
        print("first return")
        return footprint
    for idx, country in enumerate(graph[key]):
        print(idx,country)
        graph[key].pop(idx)
        tmp = footprint[:]
        tmp.append(country)
        
        ret = dfs(graph, N, country, tmp)
        
        graph[key].insert(idx, country)
        if ret:
            return ret
        
        
        
        
def solution2(tickets):
     
    graph = defaultdict(list)

    N = len(tickets)
    for ticket in tickets:
        graph[ticket[0]].append(ticket[1])
        graph[ticket[0]].sort()
    print(graph)
    
    return dfs(graph, N, "ICN", ["ICN"])


