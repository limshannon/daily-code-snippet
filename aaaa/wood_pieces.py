def is_valid_cut(wood, cur_len, k):
    pieces = 0
    for w in wood:
        pieces += w // cur_len
    return True if pieces >= k else False


def cut_wood(wood, k):
    max_len = max(wood)
    left, right = 1, max_len
    while left + 1 < right:
        mid = (left + right) // 2
        if is_valid_cut(wood, mid, k):
            left = mid
        else:
            right = mid - 1
    if is_valid_cut(wood, right, k):
        return right
    elif is_valid_cut(wood, left, k):
        return left
    else:
        return 0
