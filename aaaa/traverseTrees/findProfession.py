def solution(level, pos):
    if (level == 1): 
        return 'Engineer'
  
    # Recursively find parent's profession. If parent 
    # is a doctar, this node will be a doctal if it is 
    # at odd position and an engineer if at even position 
    if (solution(level-1, (pos+1)//2) == 'Doctor'): 
        if (pos%2): 
            return 'Doctor'
        else: 
            return 'Engineer'
  
    # If parent is an engineer, then current node will be 
    # an enginner if at add position and doctor if even 
    # position. 
    if(pos%2): 
        return 'Engineer'
    else: 
        return 'Doctor'
def solution(level, pos):
    """
    Level 1: E
    Level 2: ED
    Level 3: EDDE
    Level 4: EDDEDEED
    Level 5: EDDEDEEDDEEDEDDE 
    
    Level input is not necessary because first elements are the same
    The result is based on the count of 1's in binary representation of position-1
    If position is even, then Engineer; Else Doctor
    """
    bits  = bin(pos-1).count('1')
    if bits%2 == 0: 
        return "Engineer"
    else:
        return "Doctor"

#################################################
def solution(level, pos):
    if level == 1:
        return 'Engineer'

    parent = solution(level-1, (pos+1)/2)  
    if pos % 2 != 0:    # pos is odd
        return parent

    # pos is even, so curr node is OPPOSITE of parent
    if parent == 'Engineer':
        return 'Doctor'
    else:
        return 'Engineer'
#################################################
def solution(level, pos):
    return recur(level, pos, "Engineer")
def recur(level, pos, parent):
    if level == 1:
        return parent
    half = 2**(level - 2)
    if pos > half:
        if parent == "Doctor":
            return recur(level - 1, pos - half, "Engineer")
        if parent == "Engineer":
            return recur(level - 1, pos - half, "Doctor")
    if pos <= half:
        if parent == "Engineer":
            return recur(level - 1, pos, "Engineer")
        if parent == "Doctor":
            return recur(level - 1, pos, "Doctor")
