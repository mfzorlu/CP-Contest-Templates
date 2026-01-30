import sys

# No bisect module used

def binary_search_exact(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def lower_bound_manual(arr, target):
    """Returns first index where arr[i] >= target"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound_manual(arr, target):
    """Returns first index where arr[i] > target"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# Basic testing
if __name__ == "__main__":
    arr = [1, 2, 4, 4, 6]
    # lower bound of 4 is index 2, upper is 4
    pass
