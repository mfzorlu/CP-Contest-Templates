"""
LIS (Longest Increasing Subsequence) and LCS (Longest Common Subsequence)
Dynamic Programming Templates for Competitive Programming
Complete collection with all variations
"""

from typing import List, Tuple
from bisect import bisect_left, bisect_right
import sys

# ============= LONGEST INCREASING SUBSEQUENCE (LIS) =============

def lis_n_square(arr: List[int]) -> int:
    """
    LIS using O(n^2) DP - Classic approach
    Time: O(n^2), Space: O(n)
    """
    if not arr:
        return 0
    
    n = len(arr)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def lis_n_logn(arr: List[int]) -> int:
    """
    LIS using O(n log n) with binary search - BEST for CP
    Time: O(n log n), Space: O(n)
    
    Key idea: Maintain array of smallest tail elements for each length
    """
    if not arr:
        return 0
    
    tails = []  # tails[i] = smallest tail of all LIS with length i+1
    
    for num in arr:
        pos = bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def lis_with_sequence(arr: List[int]) -> Tuple[int, List[int]]:
    """
    LIS that returns the actual sequence
    Time: O(n^2), Space: O(n)
    Returns: (length, actual_sequence)
    """
    if not arr:
        return 0, []
    
    n = len(arr)
    dp = [1] * n
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find maximum length and its ending position
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # Reconstruct sequence
    sequence = []
    idx = max_index
    while idx != -1:
        sequence.append(arr[idx])
        idx = parent[idx]
    
    sequence.reverse()
    return max_length, sequence


def lis_n_logn_with_sequence(arr: List[int]) -> Tuple[int, List[int]]:
    """
    LIS O(n log n) with sequence reconstruction
    Time: O(n log n), Space: O(n)
    """
    if not arr:
        return 0, []
    
    n = len(arr)
    tails = []
    parent = [-1] * n
    indices = []  # indices[i] stores index of element in tails[i]
    
    for i, num in enumerate(arr):
        pos = bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
            indices.append(i)
        else:
            tails[pos] = num
            indices[pos] = i
        
        if pos > 0:
            parent[i] = indices[pos - 1]
    
    # Reconstruct sequence
    sequence = []
    idx = indices[-1]
    while idx != -1:
        sequence.append(arr[idx])
        idx = parent[idx]
    
    sequence.reverse()
    return len(tails), sequence


# ============= LIS VARIATIONS =============

def lis_non_decreasing(arr: List[int]) -> int:
    """
    Longest Non-Decreasing Subsequence (allows equal elements)
    Use bisect_right instead of bisect_left
    Time: O(n log n)
    """
    tails = []
    
    for num in arr:
        pos = bisect_right(tails, num)  # Changed from bisect_left
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def lis_strictly_decreasing(arr: List[int]) -> int:
    """
    Longest Strictly Decreasing Subsequence
    Negate array and find LIS
    Time: O(n log n)
    """
    negated = [-x for x in arr]
    return lis_n_logn(negated)


def lis_with_bounds(arr: List[int], lower: int, upper: int) -> int:
    """
    LIS where all elements must be in range [lower, upper]
    Time: O(n log n)
    """
    filtered = [x for x in arr if lower <= x <= upper]
    return lis_n_logn(filtered)


def number_of_lis(arr: List[int]) -> int:
    """
    Count number of different LIS
    Time: O(n^2), Space: O(n)
    """
    if not arr:
        return 0
    
    n = len(arr)
    lengths = [1] * n
    counts = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
    
    max_length = max(lengths)
    return sum(counts[i] for i in range(n) if lengths[i] == max_length)


def lis_k_decreasing(arr: List[int], k: int) -> int:
    """
    Longest subsequence where arr[i+1] >= arr[i] - k
    Time: O(n log n)
    """
    # Transform: add i*k to arr[i], then find LIS
    transformed = [arr[i] + i * k for i in range(len(arr))]
    return lis_n_logn(transformed)


# ============= LONGEST COMMON SUBSEQUENCE (LCS) =============

def lcs_2d(text1: str, text2: str) -> int:
    """
    Classic LCS with 2D DP
    Time: O(m * n), Space: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def lcs_1d(text1: str, text2: str) -> int:
    """
    Space-optimized LCS - BEST for CP when only length needed
    Time: O(m * n), Space: O(min(m, n))
    """
    # Make sure text1 is shorter for space optimization
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    
    return prev[n]


def lcs_with_sequence(text1: str, text2: str) -> Tuple[int, str]:
    """
    LCS that returns the actual sequence
    Time: O(m * n), Space: O(m * n)
    Returns: (length, actual_lcs_string)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return dp[m][n], ''.join(lcs)


def lcs_three_strings(s1: str, s2: str, s3: str) -> int:
    """
    LCS of three strings
    Time: O(m * n * p), Space: O(m * n * p)
    """
    m, n, p = len(s1), len(s2), len(s3)
    dp = [[[0] * (p + 1) for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            for k in range(1, p + 1):
                if s1[i-1] == s2[j-1] == s3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], 
                                     dp[i][j-1][k], 
                                     dp[i][j][k-1])
    
    return dp[m][n][p]


# ============= LCS VARIATIONS =============

def longest_palindromic_subsequence(s: str) -> int:
    """
    Longest Palindromic Subsequence = LCS(s, reverse(s))
    Time: O(n^2), Space: O(n)
    """
    return lcs_1d(s, s[::-1])


def shortest_common_supersequence_length(text1: str, text2: str) -> int:
    """
    Shortest string that has both text1 and text2 as subsequences
    SCS_length = len(text1) + len(text2) - LCS_length
    Time: O(m * n)
    """
    lcs_len = lcs_1d(text1, text2)
    return len(text1) + len(text2) - lcs_len


def minimum_insertions_to_palindrome(s: str) -> int:
    """
    Minimum insertions to make string palindrome
    = n - LCS(s, reverse(s))
    Time: O(n^2)
    """
    lcs_len = lcs_1d(s, s[::-1])
    return len(s) - lcs_len


def minimum_deletions_to_palindrome(s: str) -> int:
    """
    Minimum deletions to make string palindrome
    Same as minimum insertions
    Time: O(n^2)
    """
    return minimum_insertions_to_palindrome(s)


def lcs_of_arrays(arr1: List[int], arr2: List[int]) -> int:
    """
    LCS for integer arrays (works same as strings)
    Time: O(m * n), Space: O(n)
    """
    m, n = len(arr1), len(arr2)
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if arr1[i-1] == arr2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    
    return prev[n]


# ============= LIS + LCS COMBINED PROBLEMS =============

def longest_bitonic_subsequence(arr: List[int]) -> int:
    """
    Longest Bitonic Subsequence
    First increasing then decreasing
    Time: O(n^2)
    """
    n = len(arr)
    if n == 0:
        return 0
    
    # LIS ending at each position
    lis = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                lis[i] = max(lis[i], lis[j] + 1)
    
    # LDS starting from each position
    lds = [1] * n
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if arr[j] < arr[i]:
                lds[i] = max(lds[i], lds[j] + 1)
    
    # Maximum bitonic length
    return max(lis[i] + lds[i] - 1 for i in range(n))


def lis_from_lcs(arr: List[int]) -> int:
    """
    LIS using LCS approach (when array has duplicates)
    Create sorted unique array and find LCS
    Time: O(n^2) but handles duplicates well
    """
    sorted_arr = sorted(set(arr))
    return lcs_of_arrays(arr, sorted_arr)


# ============= QUICK CP TEMPLATES =============

def lis(arr):
    """Ultra-compact LIS O(n log n)"""
    from bisect import bisect_left
    t = []
    for x in arr:
        p = bisect_left(t, x)
        if p == len(t): t.append(x)
        else: t[p] = x
    return len(t)


def lcs(s1, s2):
    """Ultra-compact LCS"""
    m, n = len(s1), len(s2)
    p = [0] * (n + 1)
    for i in range(1, m + 1):
        c = [0] * (n + 1)
        for j in range(1, n + 1):
            c[j] = p[j-1] + 1 if s1[i-1] == s2[j-1] else max(p[j], c[j-1])
        p = c
    return p[n]


# ============= SPECIALIZED TEMPLATES =============

def lis_divide_conquer(arr: List[int]) -> int:
    """
    LIS using divide and conquer with segment tree
    Advanced technique for online queries
    Time: O(n log n)
    """
    # Coordinate compression
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    
    # Segment tree for range maximum query
    n = len(sorted_unique)
    tree = [0] * (4 * n)
    
    def update(node, start, end, idx, val):
        if start == end:
            tree[node] = max(tree[node], val)
        else:
            mid = (start + end) // 2
            if idx <= mid:
                update(2*node, start, mid, idx, val)
            else:
                update(2*node+1, mid+1, end, idx, val)
            tree[node] = max(tree[2*node], tree[2*node+1])
    
    def query(node, start, end, left, right):
        if left > end or right < start:
            return 0
        if left <= start and end <= right:
            return tree[node]
        mid = (start + end) // 2
        return max(query(2*node, start, mid, left, right),
                  query(2*node+1, mid+1, end, left, right))
    
    max_lis = 0
    for num in arr:
        idx = compress[num]
        max_before = query(1, 0, n-1, 0, idx-1) if idx > 0 else 0
        new_val = max_before + 1
        update(1, 0, n-1, idx, new_val)
        max_lis = max(max_lis, new_val)
    
    return max_lis


# ============= TESTING =============

if __name__ == "__main__":
    print("=" * 70)
    print("LIS (Longest Increasing Subsequence) Templates")
    print("=" * 70)
    
    arr = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"\nArray: {arr}")
    print(f"LIS length (O(n^2)): {lis_n_square(arr)}")
    print(f"LIS length (O(n log n)): {lis_n_logn(arr)}")
    
    length, sequence = lis_with_sequence(arr)
    print(f"LIS sequence: {sequence}, length: {length}")
    
    length2, sequence2 = lis_n_logn_with_sequence(arr)
    print(f"LIS sequence (O(n log n)): {sequence2}, length: {length2}")
    
    print(f"\nNumber of different LIS: {number_of_lis(arr)}")
    
    print("\n" + "=" * 70)
    print("LCS (Longest Common Subsequence) Templates")
    print("=" * 70)
    
    text1, text2 = "abcde", "ace"
    print(f"\nText1: {text1}")
    print(f"Text2: {text2}")
    print(f"LCS length (2D): {lcs_2d(text1, text2)}")
    print(f"LCS length (1D): {lcs_1d(text1, text2)}")
    
    length, lcs_str = lcs_with_sequence(text1, text2)
    print(f"LCS string: {lcs_str}, length: {length}")
    
    print("\n" + "=" * 70)
    print("LCS Applications")
    print("=" * 70)
    
    s = "bbbab"
    print(f"\nString: {s}")
    print(f"Longest Palindromic Subsequence: {longest_palindromic_subsequence(s)}")
    print(f"Min insertions to palindrome: {minimum_insertions_to_palindrome(s)}")
    
    print(f"\nShortest Common Supersequence length of '{text1}' and '{text2}':")
    print(f"  {shortest_common_supersequence_length(text1, text2)}")
    
    print("\n" + "=" * 70)
    print("Special Subsequences")
    print("=" * 70)
    
    arr2 = [1, 11, 2, 10, 4, 5, 2, 1]
    print(f"\nArray: {arr2}")
    print(f"Longest Bitonic Subsequence: {longest_bitonic_subsequence(arr2)}")
    
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    
    import time
    large_arr = list(range(1000, 0, -1))
    
    start = time.time()
    result1 = lis_n_square(large_arr)
    time1 = time.time() - start
    
    start = time.time()
    result2 = lis_n_logn(large_arr)
    time2 = time.time() - start
    
    print(f"\nArray size: {len(large_arr)}")
    print(f"O(n^2) algorithm: {time1:.6f}s, result: {result1}")
    print(f"O(n log n) algorithm: {time2:.6f}s, result: {result2}")
    print(f"Speedup: {time1/time2:.2f}x")


"""
// C++ Version

/*
 * LIS (Longest Increasing Subsequence) and LCS (Longest Common Subsequence)
 * Dynamic Programming Templates for Competitive Programming
 * Complete collection with all variations
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <chrono>

using namespace std;
typedef long long ll;

// ============= LONGEST INCREASING SUBSEQUENCE (LIS) =============

// O(n^2) Classic DP approach
int lis_n_square(vector<int>& arr) {
    /*
     * Classic LIS with O(n^2) DP
     * Time: O(n^2), Space: O(n)
     */
    int n = arr.size();
    if (n == 0) return 0;
    
    vector<int> dp(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    
    return *max_element(dp.begin(), dp.end());
}

// O(n log n) Binary Search approach - BEST for CP
int lis_n_logn(vector<int>& arr) {
    /*
     * LIS using O(n log n) with binary search
     * Time: O(n log n), Space: O(n)
     * 
     * Key idea: Maintain array of smallest tail elements for each length
     * tails[i] = smallest tail of all LIS with length i+1
     */
    if (arr.empty()) return 0;
    
    vector<int> tails;
    
    for (int num : arr) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return tails.size();
}

// LIS with sequence reconstruction O(n^2)
pair<int, vector<int>> lis_with_sequence(vector<int>& arr) {
    /*
     * LIS that returns the actual sequence
     * Time: O(n^2), Space: O(n)
     * Returns: {length, actual_sequence}
     */
    int n = arr.size();
    if (n == 0) return {0, {}};
    
    vector<int> dp(n, 1);
    vector<int> parent(n, -1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    
    // Find maximum length and its ending position
    int max_length = *max_element(dp.begin(), dp.end());
    int max_index = max_element(dp.begin(), dp.end()) - dp.begin();
    
    // Reconstruct sequence
    vector<int> sequence;
    int idx = max_index;
    while (idx != -1) {
        sequence.push_back(arr[idx]);
        idx = parent[idx];
    }
    
    reverse(sequence.begin(), sequence.end());
    return {max_length, sequence};
}

// LIS O(n log n) with sequence reconstruction
pair<int, vector<int>> lis_n_logn_with_sequence(vector<int>& arr) {
    /*
     * LIS O(n log n) with sequence reconstruction
     * Time: O(n log n), Space: O(n)
     */
    int n = arr.size();
    if (n == 0) return {0, {}};
    
    vector<int> tails;
    vector<int> parent(n, -1);
    vector<int> indices;  // indices[i] stores index of element in tails[i]
    
    for (int i = 0; i < n; i++) {
        int num = arr[i];
        auto it = lower_bound(tails.begin(), tails.end(), num);
        int pos = it - tails.begin();
        
        if (it == tails.end()) {
            tails.push_back(num);
            indices.push_back(i);
        } else {
            *it = num;
            indices[pos] = i;
        }
        
        if (pos > 0) {
            parent[i] = indices[pos - 1];
        }
    }
    
    // Reconstruct sequence
    vector<int> sequence;
    int idx = indices.back();
    while (idx != -1) {
        sequence.push_back(arr[idx]);
        idx = parent[idx];
    }
    
    reverse(sequence.begin(), sequence.end());
    return {(int)tails.size(), sequence};
}

// ============= LIS VARIATIONS =============

// Longest Non-Decreasing Subsequence (allows equal elements)
int lis_non_decreasing(vector<int>& arr) {
    /*
     * Use upper_bound instead of lower_bound
     * Time: O(n log n)
     */
    vector<int> tails;
    
    for (int num : arr) {
        auto it = upper_bound(tails.begin(), tails.end(), num);
        
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return tails.size();
}

// Longest Strictly Decreasing Subsequence
int lis_strictly_decreasing(vector<int>& arr) {
    /*
     * Negate array and find LIS
     * Time: O(n log n)
     */
    vector<int> negated;
    for (int x : arr) negated.push_back(-x);
    return lis_n_logn(negated);
}

// Count number of different LIS
int number_of_lis(vector<int>& arr) {
    /*
     * Count number of different LIS
     * Time: O(n^2), Space: O(n)
     */
    int n = arr.size();
    if (n == 0) return 0;
    
    vector<int> lengths(n, 1);
    vector<int> counts(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                if (lengths[j] + 1 > lengths[i]) {
                    lengths[i] = lengths[j] + 1;
                    counts[i] = counts[j];
                } else if (lengths[j] + 1 == lengths[i]) {
                    counts[i] += counts[j];
                }
            }
        }
    }
    
    int max_length = *max_element(lengths.begin(), lengths.end());
    int total_count = 0;
    for (int i = 0; i < n; i++) {
        if (lengths[i] == max_length) {
            total_count += counts[i];
        }
    }
    
    return total_count;
}

// ============= LONGEST COMMON SUBSEQUENCE (LCS) =============

// Classic LCS with 2D DP
int lcs_2d(string text1, string text2) {
    /*
     * Classic LCS with 2D DP
     * Time: O(m * n), Space: O(m * n)
     */
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    return dp[m][n];
}

// Space-optimized LCS - BEST for CP
int lcs_1d(string text1, string text2) {
    /*
     * Space-optimized LCS
     * Time: O(m * n), Space: O(min(m, n))
     */
    // Make sure text1 is shorter
    if (text1.length() > text2.length()) {
        swap(text1, text2);
    }
    
    int m = text1.length(), n = text2.length();
    vector<int> prev(n + 1, 0);
    
    for (int i = 1; i <= m; i++) {
        vector<int> curr(n + 1, 0);
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                curr[j] = prev[j-1] + 1;
            } else {
                curr[j] = max(prev[j], curr[j-1]);
            }
        }
        prev = curr;
    }
    
    return prev[n];
}

// LCS with sequence reconstruction
pair<int, string> lcs_with_sequence(string text1, string text2) {
    /*
     * LCS that returns the actual sequence
     * Time: O(m * n), Space: O(m * n)
     * Returns: {length, actual_lcs_string}
     */
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    // Reconstruct LCS
    string lcs = "";
    int i = m, j = n;
    
    while (i > 0 && j > 0) {
        if (text1[i-1] == text2[j-1]) {
            lcs = text1[i-1] + lcs;
            i--;
            j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }
    
    return {dp[m][n], lcs};
}

// LCS of three strings
int lcs_three_strings(string s1, string s2, string s3) {
    /*
     * LCS of three strings
     * Time: O(m * n * p), Space: O(m * n * p)
     */
    int m = s1.length(), n = s2.length(), p = s3.length();
    vector<vector<vector<int>>> dp(m + 1, 
        vector<vector<int>>(n + 1, vector<int>(p + 1, 0)));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            for (int k = 1; k <= p; k++) {
                if (s1[i-1] == s2[j-1] && s2[j-1] == s3[k-1]) {
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1;
                } else {
                    dp[i][j][k] = max({dp[i-1][j][k], 
                                      dp[i][j-1][k], 
                                      dp[i][j][k-1]});
                }
            }
        }
    }
    
    return dp[m][n][p];
}

// ============= LCS VARIATIONS =============

// Longest Palindromic Subsequence
int longest_palindromic_subsequence(string s) {
    /*
     * LPS = LCS(s, reverse(s))
     * Time: O(n^2), Space: O(n)
     */
    string rev = s;
    reverse(rev.begin(), rev.end());
    return lcs_1d(s, rev);
}

// Shortest Common Supersequence length
int shortest_common_supersequence_length(string text1, string text2) {
    /*
     * Shortest string that has both text1 and text2 as subsequences
     * SCS_length = len(text1) + len(text2) - LCS_length
     * Time: O(m * n)
     */
    int lcs_len = lcs_1d(text1, text2);
    return text1.length() + text2.length() - lcs_len;
}

// Minimum insertions to make palindrome
int minimum_insertions_to_palindrome(string s) {
    /*
     * Minimum insertions to make string palindrome
     * = n - LCS(s, reverse(s))
     * Time: O(n^2)
     */
    int lcs_len = longest_palindromic_subsequence(s);
    return s.length() - lcs_len;
}

// LCS for integer arrays
int lcs_of_arrays(vector<int>& arr1, vector<int>& arr2) {
    /*
     * LCS for integer arrays
     * Time: O(m * n), Space: O(n)
     */
    int m = arr1.size(), n = arr2.size();
    vector<int> prev(n + 1, 0);
    
    for (int i = 1; i <= m; i++) {
        vector<int> curr(n + 1, 0);
        for (int j = 1; j <= n; j++) {
            if (arr1[i-1] == arr2[j-1]) {
                curr[j] = prev[j-1] + 1;
            } else {
                curr[j] = max(prev[j], curr[j-1]);
            }
        }
        prev = curr;
    }
    
    return prev[n];
}

// ============= LIS + LCS COMBINED PROBLEMS =============

// Longest Bitonic Subsequence
int longest_bitonic_subsequence(vector<int>& arr) {
    /*
     * Longest Bitonic Subsequence
     * First increasing then decreasing
     * Time: O(n^2)
     */
    int n = arr.size();
    if (n == 0) return 0;
    
    // LIS ending at each position
    vector<int> lis(n, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                lis[i] = max(lis[i], lis[j] + 1);
            }
        }
    }
    
    // LDS starting from each position
    vector<int> lds(n, 1);
    for (int i = n - 2; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[i]) {
                lds[i] = max(lds[i], lds[j] + 1);
            }
        }
    }
    
    // Maximum bitonic length
    int max_length = 0;
    for (int i = 0; i < n; i++) {
        max_length = max(max_length, lis[i] + lds[i] - 1);
    }
    
    return max_length;
}

// ============= QUICK CP TEMPLATES =============

// Ultra-compact LIS
int lis(vector<int>& a) {
    vector<int> t;
    for (int x : a) {
        auto it = lower_bound(t.begin(), t.end(), x);
        if (it == t.end()) t.push_back(x);
        else *it = x;
    }
    return t.size();
}

// Ultra-compact LCS
int lcs(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    vector<int> p(n + 1, 0);
    for (int i = 1; i <= m; i++) {
        vector<int> c(n + 1, 0);
        for (int j = 1; j <= n; j++)
            c[j] = s1[i-1] == s2[j-1] ? p[j-1] + 1 : max(p[j], c[j-1]);
        p = c;
    }
    return p[n];
}

// ============= MAIN - TESTING =============

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cout << string(70, '=') << endl;
    cout << "LIS (Longest Increasing Subsequence) Templates" << endl;
    cout << string(70, '=') << endl;
    
    vector<int> arr = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "\nArray: ";
    for (int x : arr) cout << x << " ";
    cout << endl;
    
    cout << "LIS length (O(n^2)): " << lis_n_square(arr) << endl;
    cout << "LIS length (O(n log n)): " << lis_n_logn(arr) << endl;
    
    auto [length1, sequence1] = lis_with_sequence(arr);
    cout << "LIS sequence: ";
    for (int x : sequence1) cout << x << " ";
    cout << ", length: " << length1 << endl;
    
    auto [length2, sequence2] = lis_n_logn_with_sequence(arr);
    cout << "LIS sequence (O(n log n)): ";
    for (int x : sequence2) cout << x << " ";
    cout << ", length: " << length2 << endl;
    
    cout << "\nNumber of different LIS: " << number_of_lis(arr) << endl;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "LCS (Longest Common Subsequence) Templates" << endl;
    cout << string(70, '=') << endl;
    
    string text1 = "abcde", text2 = "ace";
    cout << "\nText1: " << text1 << endl;
    cout << "Text2: " << text2 << endl;
    cout << "LCS length (2D): " << lcs_2d(text1, text2) << endl;
    cout << "LCS length (1D): " << lcs_1d(text1, text2) << endl;
    
    auto [lcs_len, lcs_str] = lcs_with_sequence(text1, text2);
    cout << "LCS string: " << lcs_str << ", length: " << lcs_len << endl;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "LCS Applications" << endl;
    cout << string(70, '=') << endl;
    
    string s = "bbbab";
    cout << "\nString: " << s << endl;
    cout << "Longest Palindromic Subsequence: " 
         << longest_palindromic_subsequence(s) << endl;
    cout << "Min insertions to palindrome: " 
         << minimum_insertions_to_palindrome(s) << endl;
    
    cout << "\nShortest Common Supersequence length of '" 
         << text1 << "' and '" << text2 << "':" << endl;
    cout << "  " << shortest_common_supersequence_length(text1, text2) << endl;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "Special Subsequences" << endl;
    cout << string(70, '=') << endl;
    
    vector<int> arr2 = {1, 11, 2, 10, 4, 5, 2, 1};
    cout << "\nArray: ";
    for (int x : arr2) cout << x << " ";
    cout << endl;
    cout << "Longest Bitonic Subsequence: " 
         << longest_bitonic_subsequence(arr2) << endl;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "Performance Comparison" << endl;
    cout << string(70, '=') << endl;
    
    vector<int> large_arr;
    for (int i = 1000; i > 0; i--) large_arr.push_back(i);
    
    auto start = chrono::high_resolution_clock::now();
    int result1 = lis_n_square(large_arr);
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end - start);
    
    start = chrono::high_resolution_clock::now();
    int result2 = lis_n_logn(large_arr);
    end = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end - start);
    
    cout << "\nArray size: " << large_arr.size() << endl;
    cout << "O(n^2) algorithm: " << duration1.count() / 1000000.0 
         << "s, result: " << result1 << endl;
    cout << "O(n log n) algorithm: " << duration2.count() / 1000000.0 
         << "s, result: " << result2 << endl;
    cout << "Speedup: " << (double)duration1.count() / duration2.count() 
         << "x" << endl;
    
    return 0;
}


"""