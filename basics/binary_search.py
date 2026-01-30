"""
Binary Search Templates for Competitive Programming
Covers all binary search patterns and edge cases
Time Complexity: O(log n)
"""

import bisect
from typing import List, Callable

# ============= BASIC BINARY SEARCH =============

def binary_search_exact(arr: List[int], target: int) -> int:
    """
    Find exact position of target in sorted array
    Returns: index if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def binary_search_leftmost(arr: List[int], target: int) -> int:
    """
    Find LEFTMOST (first) occurrence of target
    Returns: index of first occurrence, or -1 if not found
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def binary_search_rightmost(arr: List[int], target: int) -> int:
    """
    Find RIGHTMOST (last) occurrence of target
    Returns: index of last occurrence, or -1 if not found
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


# ============= LOWER/UPPER BOUND =============

def lower_bound(arr: List[int], target: int) -> int:
    """
    Find first position where arr[i] >= target
    Returns: index (can be len(arr) if all elements < target)
    Same as C++ lower_bound
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def upper_bound(arr: List[int], target: int) -> int:
    """
    Find first position where arr[i] > target
    Returns: index (can be len(arr) if all elements <= target)
    Same as C++ upper_bound
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left


# ============= BINARY SEARCH ON ANSWER =============

def binary_search_answer_min(check: Callable[[int], bool], 
                              low: int, high: int) -> int:
    """
    Find MINIMUM value where check(x) is True
    Pattern: FFFF...FTTTTT
    
    Args:
        check: function that returns True/False
        low: minimum possible answer
        high: maximum possible answer
    
    Returns: minimum x where check(x) is True, or high+1 if none exists
    
    Example: Find minimum capacity to ship packages in D days
    """
    result = high + 1
    
    while low <= high:
        mid = low + (high - low) // 2
        
        if check(mid):
            result = mid
            high = mid - 1  # Try smaller
        else:
            low = mid + 1
    
    return result


def binary_search_answer_max(check: Callable[[int], bool], 
                              low: int, high: int) -> int:
    """
    Find MAXIMUM value where check(x) is True
    Pattern: TTTTT...TFFFF
    
    Args:
        check: function that returns True/False
        low: minimum possible answer
        high: maximum possible answer
    
    Returns: maximum x where check(x) is True, or low-1 if none exists
    
    Example: Find maximum distance with K pairs
    """
    result = low - 1
    
    while low <= high:
        mid = low + (high - low) // 2
        
        if check(mid):
            result = mid
            low = mid + 1  # Try larger
        else:
            high = mid - 1
    
    return result


# ============= FLOATING POINT BINARY SEARCH =============

def binary_search_float(check: Callable[[float], bool], 
                        low: float, high: float, 
                        epsilon: float = 1e-9) -> float:
    """
    Binary search on floating point values
    
    Args:
        check: function that returns True/False
        low: minimum value
        high: maximum value
        epsilon: precision (default 1e-9)
    
    Returns: answer with given precision
    """
    while high - low > epsilon:
        mid = (low + high) / 2
        
        if check(mid):
            high = mid
        else:
            low = mid
    
    return (low + high) / 2


def binary_search_float_iterations(check: Callable[[float], bool], 
                                   low: float, high: float, 
                                   iterations: int = 100) -> float:
    """
    Binary search on floats using fixed iterations
    More reliable for competitions
    """
    for _ in range(iterations):
        mid = (low + high) / 2
        
        if check(mid):
            high = mid
        else:
            low = mid
    
    return (low + high) / 2


# ============= TERNARY SEARCH =============

def ternary_search_max(f: Callable[[float], float], 
                       left: float, right: float, 
                       epsilon: float = 1e-9) -> float:
    """
    Find maximum of unimodal function
    Function must have single peak (∩ shaped)
    
    Returns: x where f(x) is maximum
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if f(mid1) < f(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2


def ternary_search_min(f: Callable[[float], float], 
                       left: float, right: float, 
                       epsilon: float = 1e-9) -> float:
    """
    Find minimum of unimodal function
    Function must have single valley (∪ shaped)
    
    Returns: x where f(x) is minimum
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if f(mid1) > f(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2


# ============= USING PYTHON BISECT MODULE =============

def bisect_examples(arr: List[int], target: int):
    """
    Python's bisect module - fastest for basic operations
    """
    # bisect_left: same as lower_bound
    pos_left = bisect.bisect_left(arr, target)
    
    # bisect_right: same as upper_bound
    pos_right = bisect.bisect_right(arr, target)
    
    # Count occurrences
    count = pos_right - pos_left
    
    # Check if target exists
    exists = pos_left < len(arr) and arr[pos_left] == target
    
    return {
        'lower_bound': pos_left,
        'upper_bound': pos_right,
        'count': count,
        'exists': exists
    }


# ============= ROTATED ARRAY SEARCH =============

def search_rotated_array(arr: List[int], target: int) -> int:
    """
    Search in rotated sorted array
    Example: [4,5,6,7,0,1,2] rotated at index 4
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Determine which half is sorted
        if arr[left] <= arr[mid]:  # Left half is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def find_rotation_point(arr: List[int]) -> int:
    """
    Find the index of minimum element (rotation point)
    in rotated sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    
    return left


# ============= 2D BINARY SEARCH =============

def search_2d_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Search in 2D matrix where:
    - Each row is sorted
    - First element of each row > last element of previous row
    
    Treat as 1D sorted array
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_value = matrix[mid // n][mid % n]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


# ============= PRACTICAL CP EXAMPLES =============

def find_kth_smallest_pair_distance(nums: List[int], k: int) -> int:
    """
    Find k-th smallest distance among all pairs
    Classic binary search on answer problem
    """
    nums.sort()
    
    def count_pairs(mid: int) -> int:
        """Count pairs with distance <= mid"""
        count = 0
        left = 0
        for right in range(len(nums)):
            while nums[right] - nums[left] > mid:
                left += 1
            count += right - left
        return count
    
    return binary_search_answer_min(
        lambda x: count_pairs(x) >= k,
        0, nums[-1] - nums[0]
    )


def split_array_largest_sum(nums: List[int], k: int) -> int:
    """
    Split array into k subarrays to minimize largest sum
    Binary search on answer
    """
    def can_split(max_sum: int) -> bool:
        """Check if we can split into k parts with max sum <= max_sum"""
        current_sum = 0
        splits = 1
        
        for num in nums:
            if current_sum + num > max_sum:
                splits += 1
                current_sum = num
                if splits > k:
                    return False
            else:
                current_sum += num
        
        return True
    
    return binary_search_answer_min(
        can_split,
        max(nums), sum(nums)
    )


# ============= QUICK CP TEMPLATE =============

def bs_lower(arr, target):
    """Quick lower_bound for contests"""
    l, r = 0, len(arr)
    while l < r:
        m = (l + r) // 2
        if arr[m] < target: l = m + 1
        else: r = m
    return l

def bs_upper(arr, target):
    """Quick upper_bound for contests"""
    l, r = 0, len(arr)
    while l < r:
        m = (l + r) // 2
        if arr[m] <= target: l = m + 1
        else: r = m
    return l

def bs_answer_min(check, lo, hi):
    """Quick BS on answer (find minimum)"""
    res = hi + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(mid):
            res = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return res

def bs_answer_max(check, lo, hi):
    """Quick BS on answer (find maximum)"""
    res = lo - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(mid):
            res = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return res


# ============= TESTING =============

if __name__ == "__main__":
    # Test basic binary search
    arr = [1, 2, 2, 2, 3, 4, 5, 5, 5, 6]
    print("Array:", arr)
    print(f"Exact search for 5: {binary_search_exact(arr, 5)}")
    print(f"Leftmost 5: {binary_search_leftmost(arr, 5)}")
    print(f"Rightmost 5: {binary_search_rightmost(arr, 5)}")
    print(f"Lower bound of 5: {lower_bound(arr, 5)}")
    print(f"Upper bound of 5: {upper_bound(arr, 5)}")
    
    # Using bisect
    print(f"\nBisect results: {bisect_examples(arr, 5)}")
    
    # Rotated array
    rotated = [4, 5, 6, 7, 0, 1, 2]
    print(f"\nRotated array: {rotated}")
    print(f"Search for 0: {search_rotated_array(rotated, 0)}")
    print(f"Rotation point: {find_rotation_point(rotated)}")
    
    # Binary search on answer
    print("\n--- Binary Search on Answer ---")
    # Example: Find minimum capacity to ship packages
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    days = 5
    
    def can_ship(capacity):
        day_count = 1
        current_load = 0
        for w in weights:
            if w > capacity:
                return False
            if current_load + w > capacity:
                day_count += 1
                current_load = w
            else:
                current_load += w
        return day_count <= days
    
    min_capacity = binary_search_answer_min(can_ship, 1, sum(weights))
    print(f"Minimum capacity to ship in {days} days: {min_capacity}")
    
    # Ternary search example
    print("\n--- Ternary Search ---")
    # Find maximum of -x^2 + 4x + 3
    f = lambda x: -x**2 + 4*x + 3
    max_x = ternary_search_max(f, -10, 10)
    print(f"Maximum at x = {max_x:.6f}, f(x) = {f(max_x):.6f}")


"""
// C++ Version

/*
 * Binary Search Templates for Competitive Programming
 * Covers all binary search patterns and edge cases
 * Time Complexity: O(log n)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

using namespace std;
typedef long long ll;

// ============= BASIC BINARY SEARCH =============

// Find exact position of target in sorted array
int binary_search_exact(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;  // Avoid overflow
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;  // Not found
}

// Find LEFTMOST (first) occurrence of target
int binary_search_leftmost(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// Find RIGHTMOST (last) occurrence of target
int binary_search_rightmost(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            left = mid + 1;  // Continue searching right
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// ============= LOWER/UPPER BOUND =============

// Find first position where arr[i] >= target (same as C++ lower_bound)
int lower_bound_custom(vector<int>& arr, int target) {
    int left = 0, right = arr.size();
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

// Find first position where arr[i] > target (same as C++ upper_bound)
int upper_bound_custom(vector<int>& arr, int target) {
    int left = 0, right = arr.size();
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

// ============= BINARY SEARCH ON ANSWER =============

// Find MINIMUM value where check(x) is True
// Pattern: FFFF...FTTTTT
template<typename F>
ll binary_search_answer_min(F check, ll low, ll high) {
    ll result = high + 1;
    
    while (low <= high) {
        ll mid = low + (high - low) / 2;
        
        if (check(mid)) {
            result = mid;
            high = mid - 1;  // Try smaller
        } else {
            low = mid + 1;
        }
    }
    
    return result;
}

// Find MAXIMUM value where check(x) is True
// Pattern: TTTTT...TFFFF
template<typename F>
ll binary_search_answer_max(F check, ll low, ll high) {
    ll result = low - 1;
    
    while (low <= high) {
        ll mid = low + (high - low) / 2;
        
        if (check(mid)) {
            result = mid;
            low = mid + 1;  // Try larger
        } else {
            high = mid - 1;
        }
    }
    
    return result;
}

// ============= FLOATING POINT BINARY SEARCH =============

// Binary search on floating point values using epsilon
template<typename F>
double binary_search_float(F check, double low, double high, double epsilon = 1e-9) {
    while (high - low > epsilon) {
        double mid = (low + high) / 2.0;
        
        if (check(mid)) {
            high = mid;
        } else {
            low = mid;
        }
    }
    
    return (low + high) / 2.0;
}

// Binary search on floats using fixed iterations (more reliable)
template<typename F>
double binary_search_float_iterations(F check, double low, double high, int iterations = 100) {
    for (int i = 0; i < iterations; i++) {
        double mid = (low + high) / 2.0;
        
        if (check(mid)) {
            high = mid;
        } else {
            low = mid;
        }
    }
    
    return (low + high) / 2.0;
}

// ============= TERNARY SEARCH =============

// Find maximum of unimodal function (∩ shaped)
template<typename F>
double ternary_search_max(F f, double left, double right, double epsilon = 1e-9) {
    while (right - left > epsilon) {
        double mid1 = left + (right - left) / 3;
        double mid2 = right - (right - left) / 3;
        
        if (f(mid1) < f(mid2)) {
            left = mid1;
        } else {
            right = mid2;
        }
    }
    
    return (left + right) / 2.0;
}

// Find minimum of unimodal function (∪ shaped)
template<typename F>
double ternary_search_min(F f, double left, double right, double epsilon = 1e-9) {
    while (right - left > epsilon) {
        double mid1 = left + (right - left) / 3;
        double mid2 = right - (right - left) / 3;
        
        if (f(mid1) > f(mid2)) {
            left = mid1;
        } else {
            right = mid2;
        }
    }
    
    return (left + right) / 2.0;
}

// ============= USING STL LOWER_BOUND / UPPER_BOUND =============

void stl_binary_search_examples(vector<int>& arr, int target) {
    // lower_bound: first element >= target
    auto lb = lower_bound(arr.begin(), arr.end(), target);
    int lb_index = lb - arr.begin();
    
    // upper_bound: first element > target
    auto ub = upper_bound(arr.begin(), arr.end(), target);
    int ub_index = ub - arr.begin();
    
    // Count occurrences
    int count = ub_index - lb_index;
    
    // Check if exists
    bool exists = (lb != arr.end() && *lb == target);
    
    cout << "Lower bound index: " << lb_index << "\n";
    cout << "Upper bound index: " << ub_index << "\n";
    cout << "Count: " << count << "\n";
    cout << "Exists: " << (exists ? "Yes" : "No") << "\n";
}

// ============= ROTATED ARRAY SEARCH =============

// Search in rotated sorted array
int search_rotated_array(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        // Determine which half is sorted
        if (arr[left] <= arr[mid]) {  // Left half is sorted
            if (arr[left] <= target && target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {  // Right half is sorted
            if (arr[mid] < target && target <= arr[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}

// Find rotation point (index of minimum element)
int find_rotation_point(vector<int>& arr) {
    int left = 0, right = arr.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] > arr[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

// ============= 2D BINARY SEARCH =============

// Search in 2D matrix (treated as 1D sorted array)
bool search_2d_matrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int m = matrix.size(), n = matrix[0].size();
    int left = 0, right = m * n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int mid_value = matrix[mid / n][mid % n];
        
        if (mid_value == target) {
            return true;
        } else if (mid_value < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return false;
}

// ============= PRACTICAL CP EXAMPLES =============

// Find k-th smallest pair distance
int find_kth_smallest_pair_distance(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    
    auto count_pairs = [&](int mid) {
        int count = 0;
        int left = 0;
        for (int right = 0; right < nums.size(); right++) {
            while (nums[right] - nums[left] > mid) {
                left++;
            }
            count += right - left;
        }
        return count;
    };
    
    return binary_search_answer_min(
        [&](int x) { return count_pairs(x) >= k; },
        0, nums.back() - nums.front()
    );
}

// Split array to minimize largest sum
int split_array_largest_sum(vector<int>& nums, int k) {
    auto can_split = [&](ll max_sum) {
        ll current_sum = 0;
        int splits = 1;
        
        for (int num : nums) {
            if (num > max_sum) return false;
            
            if (current_sum + num > max_sum) {
                splits++;
                current_sum = num;
                if (splits > k) return false;
            } else {
                current_sum += num;
            }
        }
        
        return true;
    };
    
    ll max_val = *max_element(nums.begin(), nums.end());
    ll sum_val = accumulate(nums.begin(), nums.end(), 0LL);
    
    return binary_search_answer_min(can_split, max_val, sum_val);
}

// Minimum days to make M bouquets (LC 1482)
int min_days_bouquets(vector<int>& bloomDay, int m, int k) {
    if ((ll)m * k > bloomDay.size()) return -1;
    
    auto can_make = [&](int day) {
        int bouquets = 0;
        int flowers = 0;
        
        for (int bloom : bloomDay) {
            if (bloom <= day) {
                flowers++;
                if (flowers == k) {
                    bouquets++;
                    flowers = 0;
                }
            } else {
                flowers = 0;
            }
        }
        
        return bouquets >= m;
    };
    
    int min_day = *min_element(bloomDay.begin(), bloomDay.end());
    int max_day = *max_element(bloomDay.begin(), bloomDay.end());
    
    return binary_search_answer_min(can_make, min_day, max_day);
}

// ============= QUICK CP TEMPLATES =============

// Ultra-compact templates for contests

// Lower bound
int lb(vector<int>& a, int x) {
    int l = 0, r = a.size();
    while (l < r) {
        int m = (l + r) / 2;
        if (a[m] < x) l = m + 1;
        else r = m;
    }
    return l;
}

// Upper bound
int ub(vector<int>& a, int x) {
    int l = 0, r = a.size();
    while (l < r) {
        int m = (l + r) / 2;
        if (a[m] <= x) l = m + 1;
        else r = m;
    }
    return l;
}

// Binary search on answer (find minimum)
template<typename F>
ll bs_min(F check, ll lo, ll hi) {
    ll res = hi + 1;
    while (lo <= hi) {
        ll mid = (lo + hi) / 2;
        if (check(mid)) res = mid, hi = mid - 1;
        else lo = mid + 1;
    }
    return res;
}

// Binary search on answer (find maximum)
template<typename F>
ll bs_max(F check, ll lo, ll hi) {
    ll res = lo - 1;
    while (lo <= hi) {
        ll mid = (lo + hi) / 2;
        if (check(mid)) res = mid, lo = mid + 1;
        else hi = mid - 1;
    }
    return res;
}

// ============= MAIN - TESTING =============

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Test basic binary search
    vector<int> arr = {1, 2, 2, 2, 3, 4, 5, 5, 5, 6};
    cout << "Array: ";
    for (int x : arr) cout << x << " ";
    cout << "\n\n";
    
    int target = 5;
    cout << "Exact search for " << target << ": " << binary_search_exact(arr, target) << "\n";
    cout << "Leftmost " << target << ": " << binary_search_leftmost(arr, target) << "\n";
    cout << "Rightmost " << target << ": " << binary_search_rightmost(arr, target) << "\n";
    cout << "Lower bound of " << target << ": " << lower_bound_custom(arr, target) << "\n";
    cout << "Upper bound of " << target << ": " << upper_bound_custom(arr, target) << "\n\n";
    
    // STL examples
    cout << "STL Binary Search:\n";
    stl_binary_search_examples(arr, target);
    cout << "\n";
    
    // Rotated array
    vector<int> rotated = {4, 5, 6, 7, 0, 1, 2};
    cout << "Rotated array: ";
    for (int x : rotated) cout << x << " ";
    cout << "\n";
    cout << "Search for 0: " << search_rotated_array(rotated, 0) << "\n";
    cout << "Rotation point: " << find_rotation_point(rotated) << "\n\n";
    
    // Binary search on answer
    cout << "--- Binary Search on Answer ---\n";
    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int days = 5;
    
    auto can_ship = [&](ll capacity) {
        int day_count = 1;
        ll current_load = 0;
        
        for (int w : weights) {
            if (w > capacity) return false;
            
            if (current_load + w > capacity) {
                day_count++;
                current_load = w;
            } else {
                current_load += w;
            }
        }
        
        return day_count <= days;
    };
    
    ll total = accumulate(weights.begin(), weights.end(), 0LL);
    ll min_capacity = binary_search_answer_min(can_ship, 1LL, total);
    cout << "Minimum capacity to ship in " << days << " days: " << min_capacity << "\n\n";
    
    // Ternary search
    cout << "--- Ternary Search ---\n";
    auto f = [](double x) { return -x*x + 4*x + 3; };
    double max_x = ternary_search_max(f, -10.0, 10.0);
    cout << "Maximum at x = " << max_x << ", f(x) = " << f(max_x) << "\n";
    
    return 0;
}



"""