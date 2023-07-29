def solution(inputstr):
    length = len(inputstr)
    if length ==1 : return True
    #isEven = False if length%2 == 1 else True
    
    list1 = list(inputstr)
    for i in range(0, (length//2)):
        
        if list1[i] != list1[-(i+1)]:
            return False
    return True
        

