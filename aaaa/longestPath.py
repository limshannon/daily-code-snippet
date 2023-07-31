def solution(fileSystem):
    maxlen = 0
    path_dict = {0:0}
    
    for file in fileSystem.split("\f"):
        depth = file.count('\t')
        name = len(file) - depth
        if '.' in file :
            maxlen = max(maxlen, path_dict[depth] + name)
        else:
            path_dict[depth+1] = path_dict[depth] + name + 1
    return maxlen
