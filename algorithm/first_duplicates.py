def solution(a):

    d = {}
    found = 0

    for i in range(len(a)):
        print(a[i])
        if a[i] in d:
            d[a[i]] += 1
        else:
            d[a[i]] = 1

        if(d[a[i]] == 2):
            return a[i]

    return -1

def solution(a):
    mySet=set()
    for el in a:
        if el in mySet:
            return el
        mySet.add(el)
    return -1
