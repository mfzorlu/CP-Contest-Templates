/*
 * Binary Search Templates for Competitive Programming
 * Covers all binary search patterns and edge cases
 * Time Complexity: O(log n)
 */

#include <bits/stdc++.h>

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
