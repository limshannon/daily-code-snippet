def solution(arr):

    if len(arr) < 2:
        return 0

    count = 0
    streak = 0
    prev_increasing = None

    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            prev_increasing = None
            streak = 0
        else:
            curr_increasing = arr[i] > arr[i-1]
            if curr_increasing != prev_increasing:
                # keep track of streak of flips between increasing and decreasing
                streak += 1
                prev_increasing = curr_increasing
            else:
                # when we break out of a streak, we reset the streak counter to 1
                streak = 1

            # number of sawtooth contiguous subarrays goes up by the current streak
            count += streak

    return count
