def solution(inputstr):
    length = len(inputstr)
    if length ==1 : return True
    #isEven = False if length%2 == 1 else True
    
    list1 = list(inputstr)
    for i in range(0, (length//2)):
        
        if list1[i] != list1[-(i+1)]:
            return False
    return True
        

#############
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    i = 1
    while left < right:
        left = left + 1
        right = right - 1
        i = i + 1
    print("Loop terminated with left = ", left, ", right = ", right, sep="")
    return ("The pointers have either reached the same index, or have crossed each other, hence we don't need to look further.")
