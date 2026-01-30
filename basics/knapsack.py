"""
Knapsack, DP, and Greedy Algorithms for Competitive Programming
Complete collection of optimization problems
"""

from typing import List, Tuple
from collections import defaultdict
import sys

# ============= 0/1 KNAPSACK (Classic DP) =============

def knapsack_01_2d(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Classic 0/1 Knapsack with 2D DP
    Time: O(n * W), Space: O(n * W)
    Each item can be taken 0 or 1 times
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 (if fits)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]


def knapsack_01_1d(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Space-optimized 0/1 Knapsack
    Time: O(n * W), Space: O(W)
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_01_with_items(weights: List[int], values: List[int], 
                           capacity: int) -> Tuple[int, List[int]]:
    """
    0/1 Knapsack that returns selected items
    Returns: (max_value, list of selected item indices)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)
            w -= weights[i-1]
    
    selected.reverse()
    return dp[n][capacity], selected


# ============= UNBOUNDED KNAPSACK =============

def knapsack_unbounded(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Unbounded Knapsack - can take unlimited copies of each item
    Time: O(n * W), Space: O(W)
    """
    dp = [0] * (capacity + 1)
    
    for w in range(capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_unbounded_alt(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Alternative unbounded knapsack (item-first loop)
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


# ============= FRACTIONAL KNAPSACK (GREEDY) =============

def knapsack_fractional(weights: List[float], values: List[float], 
                        capacity: float) -> float:
    """
    Fractional Knapsack - can take fractions of items
    Time: O(n log n), Space: O(n)
    Uses GREEDY approach (sort by value/weight ratio)
    """
    n = len(weights)
    
    # Create (value/weight, weight, value) tuples
    items = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
    
    # Sort by value/weight ratio (descending)
    items.sort(reverse=True)
    
    total_value = 0.0
    remaining_capacity = capacity
    
    for ratio, weight, value in items:
        if weight <= remaining_capacity:
            # Take whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction
            total_value += ratio * remaining_capacity
            break
    
    return total_value


# ============= BOUNDED KNAPSACK =============

def knapsack_bounded(weights: List[int], values: List[int], 
                     counts: List[int], capacity: int) -> int:
    """
    Bounded Knapsack - each item can be taken at most count[i] times
    Time: O(n * W * max(count)), Space: O(W)
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for _ in range(counts[i]):
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_bounded_optimized(weights: List[int], values: List[int], 
                               counts: List[int], capacity: int) -> int:
    """
    Optimized Bounded Knapsack using binary representation
    Time: O(W * sum(log(count))), Space: O(W)
    """
    new_weights = []
    new_values = []
    
    # Convert to 0/1 knapsack using binary representation
    for i in range(len(weights)):
        k = 1
        remaining = counts[i]
        
        while k <= remaining:
            new_weights.append(weights[i] * k)
            new_values.append(values[i] * k)
            remaining -= k
            k *= 2
        
        if remaining > 0:
            new_weights.append(weights[i] * remaining)
            new_values.append(values[i] * remaining)
    
    return knapsack_01_1d(new_weights, new_values, capacity)


# ============= SUBSET SUM (Special Knapsack) =============

def subset_sum(nums: List[int], target: int) -> bool:
    """
    Check if there's a subset with sum = target
    Time: O(n * target), Space: O(target)
    """
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]


def count_subset_sum(nums: List[int], target: int) -> int:
    """
    Count number of subsets with sum = target
    Time: O(n * target), Space: O(target)
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[target]


def partition_equal_subset(nums: List[int]) -> bool:
    """
    Check if array can be partitioned into two equal sum subsets
    Time: O(n * sum), Space: O(sum)
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    return subset_sum(nums, total // 2)


# ============= CLASSIC DP PROBLEMS =============

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LCS - Classic DP
    Time: O(m * n), Space: O(min(m, n))
    """
    if len(text1) < len(text2):
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


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    LIS - O(n log n) using binary search
    Time: O(n log n), Space: O(n)
    """
    from bisect import bisect_left
    
    tails = []
    
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def edit_distance(word1: str, word2: str) -> int:
    """
    Edit Distance (Levenshtein Distance)
    Time: O(m * n), Space: O(n)
    """
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    
    return prev[n]


def coin_change(coins: List[int], amount: int) -> int:
    """
    Minimum coins to make amount (Unbounded Knapsack variant)
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways(coins: List[int], amount: int) -> int:
    """
    Number of ways to make amount (Unbounded Knapsack variant)
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    
    return dp[amount]


# ============= CLASSIC GREEDY PROBLEMS =============

def activity_selection(start: List[int], end: List[int]) -> List[int]:
    """
    Maximum number of non-overlapping activities
    Time: O(n log n), Space: O(n)
    """
    # Create activities with (end, start, index)
    activities = sorted([(end[i], start[i], i) for i in range(len(start))])
    
    selected = [activities[0][2]]
    last_end = activities[0][0]
    
    for i in range(1, len(activities)):
        if activities[i][1] >= last_end:  # Start >= last end
            selected.append(activities[i][2])
            last_end = activities[i][0]
    
    return selected


def job_sequencing(jobs: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    """
    Job Sequencing with Deadlines
    jobs: [(job_id, deadline, profit)]
    Returns: (number_of_jobs, total_profit)
    Time: O(n^2), Space: O(max_deadline)
    """
    # Sort by profit (descending)
    jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
    
    max_deadline = max(job[1] for job in jobs)
    slots = [-1] * (max_deadline + 1)
    
    count = 0
    profit = 0
    
    for job_id, deadline, prof in jobs:
        # Find a free slot from deadline backwards
        for j in range(deadline, 0, -1):
            if slots[j] == -1:
                slots[j] = job_id
                count += 1
                profit += prof
                break
    
    return count, profit


def huffman_encoding(freq: List[Tuple[str, int]]) -> dict:
    """
    Huffman Encoding - Greedy approach
    Time: O(n log n), Space: O(n)
    """
    import heapq
    
    if len(freq) == 1:
        return {freq[0][0]: '0'}
    
    # Create min heap
    heap = [[weight, [symbol, ""]] for symbol, weight in freq]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return dict(heap[0][1:])


def minimum_platforms(arrival: List[int], departure: List[int]) -> int:
    """
    Minimum railway platforms needed
    Time: O(n log n), Space: O(1)
    """
    arrival.sort()
    departure.sort()
    
    platforms_needed = 0
    max_platforms = 0
    i = j = 0
    
    while i < len(arrival):
        if arrival[i] <= departure[j]:
            platforms_needed += 1
            i += 1
            max_platforms = max(max_platforms, platforms_needed)
        else:
            platforms_needed -= 1
            j += 1
    
    return max_platforms


# ============= ADVANCED DP PATTERNS =============

def matrix_chain_multiplication(dims: List[int]) -> int:
    """
    Minimum scalar multiplications for matrix chain
    dims: [d0, d1, d2, ...] where matrix i has dimensions d[i-1] x d[i]
    Time: O(n^3), Space: O(n^2)
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dims[i] * dims[k+1] * dims[j+1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]


def rod_cutting(prices: List[int], n: int) -> int:
    """
    Maximum profit from cutting rod of length n
    Time: O(n^2), Space: O(n)
    """
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            if j <= len(prices):
                dp[i] = max(dp[i], prices[j-1] + dp[i-j])
    
    return dp[n]


def egg_drop(eggs: int, floors: int) -> int:
    """
    Minimum trials in worst case with k eggs and n floors
    Time: O(eggs * floors^2), Space: O(eggs * floors)
    """
    dp = [[float('inf')] * (floors + 1) for _ in range(eggs + 1)]
    
    # Base cases
    for i in range(eggs + 1):
        dp[i][0] = 0
        dp[i][1] = 1
    
    for j in range(floors + 1):
        dp[1][j] = j
    
    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            for k in range(1, j + 1):
                # Egg breaks or doesn't break
                worst = 1 + max(dp[i-1][k-1], dp[i][j-k])
                dp[i][j] = min(dp[i][j], worst)
    
    return dp[eggs][floors]


# ============= QUICK CP TEMPLATES =============

def ks01(w, v, W):
    """Quick 0/1 Knapsack"""
    dp = [0] * (W + 1)
    for i in range(len(w)):
        for j in range(W, w[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    return dp[W]


def ks_unb(w, v, W):
    """Quick Unbounded Knapsack"""
    dp = [0] * (W + 1)
    for i in range(len(w)):
        for j in range(w[i], W + 1):
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    return dp[W]


def lcs(s1, s2):
    """Quick LCS"""
    m, n = len(s1), len(s2)
    p = [0] * (n + 1)
    for i in range(1, m + 1):
        c = [0] * (n + 1)
        for j in range(1, n + 1):
            c[j] = p[j-1] + 1 if s1[i-1] == s2[j-1] else max(p[j], c[j-1])
        p = c
    return p[n]


# ============= TESTING =============

if __name__ == "__main__":
    print("=== 0/1 Knapsack ===")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    print(f"Max value: {knapsack_01_1d(weights, values, capacity)}")
    
    max_val, items = knapsack_01_with_items(weights, values, capacity)
    print(f"Selected items: {items}, Value: {max_val}")
    
    print("\n=== Unbounded Knapsack ===")
    print(f"Max value: {knapsack_unbounded(weights, values, capacity)}")
    
    print("\n=== Fractional Knapsack ===")
    weights_f = [10.0, 20.0, 30.0]
    values_f = [60.0, 100.0, 120.0]
    capacity_f = 50.0
    print(f"Max value: {knapsack_fractional(weights_f, values_f, capacity_f):.2f}")
    
    print("\n=== Subset Sum ===")
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    print(f"Subset with sum {target} exists: {subset_sum(nums, target)}")
    print(f"Number of subsets: {count_subset_sum(nums, target)}")
    
    print("\n=== Coin Change ===")
    coins = [1, 2, 5]
    amount = 11
    print(f"Minimum coins: {coin_change(coins, amount)}")
    print(f"Number of ways: {coin_change_ways(coins, amount)}")
    
    print("\n=== LCS ===")
    text1, text2 = "abcde", "ace"
    print(f"LCS length: {longest_common_subsequence(text1, text2)}")
    
    print("\n=== LIS ===")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"LIS length: {longest_increasing_subsequence(nums)}")
    
    print("\n=== Activity Selection ===")
    start = [1, 3, 0, 5, 8, 5]
    end = [2, 4, 6, 7, 9, 9]
    selected = activity_selection(start, end)
    print(f"Selected activities: {selected}")
    
    print("\n=== Job Sequencing ===")
    jobs = [(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)]
    count, profit = job_sequencing(jobs)
    print(f"Jobs: {count}, Total profit: {profit}")


"""
// C++ Version

/*
 * Knapsack, DP, and Greedy Algorithms for Competitive Programming
 * Complete collection of optimization problems
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <cstring>
#include <climits>

using namespace std;
typedef long long ll;

// ============= 0/1 KNAPSACK (Classic DP) =============

// Classic 0/1 Knapsack with 2D DP
int knapsack_01_2d(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            dp[i][w] = dp[i-1][w];  // Don't take
            
            if (weights[i-1] <= w) {  // Take if fits
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }
    
    return dp[n][capacity];
}

// Space-optimized 0/1 Knapsack (1D DP)
int knapsack_01_1d(vector<int>& weights, vector<int>& values, int capacity) {
    vector<int> dp(capacity + 1, 0);
    
    for (int i = 0; i < weights.size(); i++) {
        // Traverse backwards to avoid using updated values
        for (int w = capacity; w >= weights[i]; w--) {
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }
    
    return dp[capacity];
}

// 0/1 Knapsack with items tracking
pair<int, vector<int>> knapsack_01_with_items(vector<int>& weights, 
                                              vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    // Fill DP table
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            dp[i][w] = dp[i-1][w];
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }
    
    // Backtrack to find selected items
    vector<int> selected;
    int w = capacity;
    for (int i = n; i > 0; i--) {
        if (dp[i][w] != dp[i-1][w]) {
            selected.push_back(i - 1);
            w -= weights[i-1];
        }
    }
    
    reverse(selected.begin(), selected.end());
    return {dp[n][capacity], selected};
}

// ============= UNBOUNDED KNAPSACK =============

// Unbounded Knapsack - unlimited copies of each item
int knapsack_unbounded(vector<int>& weights, vector<int>& values, int capacity) {
    vector<int> dp(capacity + 1, 0);
    
    for (int w = 0; w <= capacity; w++) {
        for (int i = 0; i < weights.size(); i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }
    
    return dp[capacity];
}

// Alternative unbounded knapsack (item-first loop)
int knapsack_unbounded_alt(vector<int>& weights, vector<int>& values, int capacity) {
    vector<int> dp(capacity + 1, 0);
    
    for (int i = 0; i < weights.size(); i++) {
        for (int w = weights[i]; w <= capacity; w++) {
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }
    
    return dp[capacity];
}

// ============= FRACTIONAL KNAPSACK (GREEDY) =============

double knapsack_fractional(vector<double>& weights, vector<double>& values, 
                          double capacity) {
    int n = weights.size();
    
    // Create (ratio, weight, value) tuples
    vector<tuple<double, double, double>> items;
    for (int i = 0; i < n; i++) {
        items.push_back({values[i] / weights[i], weights[i], values[i]});
    }
    
    // Sort by ratio (descending)
    sort(items.begin(), items.end(), greater<tuple<double, double, double>>());
    
    double total_value = 0.0;
    double remaining_capacity = capacity;
    
    for (auto [ratio, weight, value] : items) {
        if (weight <= remaining_capacity) {
            total_value += value;
            remaining_capacity -= weight;
        } else {
            total_value += ratio * remaining_capacity;
            break;
        }
    }
    
    return total_value;
}

// ============= BOUNDED KNAPSACK =============

// Bounded Knapsack - each item can be taken at most count[i] times
int knapsack_bounded(vector<int>& weights, vector<int>& values, 
                    vector<int>& counts, int capacity) {
    vector<int> dp(capacity + 1, 0);
    
    for (int i = 0; i < weights.size(); i++) {
        for (int k = 0; k < counts[i]; k++) {
            for (int w = capacity; w >= weights[i]; w--) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }
    
    return dp[capacity];
}

// Optimized Bounded Knapsack using binary representation
int knapsack_bounded_optimized(vector<int>& weights, vector<int>& values, 
                              vector<int>& counts, int capacity) {
    vector<int> new_weights, new_values;
    
    // Convert to 0/1 knapsack using binary representation
    for (int i = 0; i < weights.size(); i++) {
        int k = 1;
        int remaining = counts[i];
        
        while (k <= remaining) {
            new_weights.push_back(weights[i] * k);
            new_values.push_back(values[i] * k);
            remaining -= k;
            k *= 2;
        }
        
        if (remaining > 0) {
            new_weights.push_back(weights[i] * remaining);
            new_values.push_back(values[i] * remaining);
        }
    }
    
    return knapsack_01_1d(new_weights, new_values, capacity);
}

// ============= SUBSET SUM (Special Knapsack) =============

// Check if there's a subset with sum = target
bool subset_sum(vector<int>& nums, int target) {
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    
    for (int num : nums) {
        for (int s = target; s >= num; s--) {
            dp[s] = dp[s] || dp[s - num];
        }
    }
    
    return dp[target];
}

// Count number of subsets with sum = target
int count_subset_sum(vector<int>& nums, int target) {
    vector<int> dp(target + 1, 0);
    dp[0] = 1;
    
    for (int num : nums) {
        for (int s = target; s >= num; s--) {
            dp[s] += dp[s - num];
        }
    }
    
    return dp[target];
}

// Check if array can be partitioned into two equal sum subsets
bool partition_equal_subset(vector<int>& nums) {
    int total = 0;
    for (int num : nums) total += num;
    
    if (total % 2 != 0) return false;
    
    return subset_sum(nums, total / 2);
}

// ============= CLASSIC DP PROBLEMS =============

// Longest Common Subsequence
int longest_common_subsequence(string text1, string text2) {
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

// Longest Increasing Subsequence - O(n log n)
int longest_increasing_subsequence(vector<int>& nums) {
    vector<int> tails;
    
    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return tails.size();
}

// Edit Distance (Levenshtein Distance)
int edit_distance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<int> prev(n + 1);
    
    for (int j = 0; j <= n; j++) prev[j] = j;
    
    for (int i = 1; i <= m; i++) {
        vector<int> curr(n + 1);
        curr[0] = i;
        
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = 1 + min({prev[j], curr[j-1], prev[j-1]});
            }
        }
        
        prev = curr;
    }
    
    return prev[n];
}

// Coin Change - Minimum coins to make amount
int coin_change(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    
    for (int coin : coins) {
        for (int x = coin; x <= amount; x++) {
            if (dp[x - coin] != INT_MAX) {
                dp[x] = min(dp[x], dp[x - coin] + 1);
            }
        }
    }
    
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}

// Coin Change - Number of ways to make amount
int coin_change_ways(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int coin : coins) {
        for (int x = coin; x <= amount; x++) {
            dp[x] += dp[x - coin];
        }
    }
    
    return dp[amount];
}

// ============= CLASSIC GREEDY PROBLEMS =============

// Activity Selection - Maximum non-overlapping activities
vector<int> activity_selection(vector<int>& start, vector<int>& end) {
    int n = start.size();
    vector<tuple<int, int, int>> activities;  // (end, start, index)
    
    for (int i = 0; i < n; i++) {
        activities.push_back({end[i], start[i], i});
    }
    
    sort(activities.begin(), activities.end());
    
    vector<int> selected;
    selected.push_back(get<2>(activities[0]));
    int last_end = get<0>(activities[0]);
    
    for (int i = 1; i < n; i++) {
        if (get<1>(activities[i]) >= last_end) {
            selected.push_back(get<2>(activities[i]));
            last_end = get<0>(activities[i]);
        }
    }
    
    return selected;
}

// Job Sequencing with Deadlines
pair<int, int> job_sequencing(vector<tuple<int, int, int>>& jobs) {
    // Sort by profit (descending)
    sort(jobs.begin(), jobs.end(), 
         [](auto& a, auto& b) { return get<2>(a) > get<2>(b); });
    
    int max_deadline = 0;
    for (auto& [id, deadline, profit] : jobs) {
        max_deadline = max(max_deadline, deadline);
    }
    
    vector<int> slots(max_deadline + 1, -1);
    int count = 0, profit = 0;
    
    for (auto& [id, deadline, prof] : jobs) {
        for (int j = deadline; j > 0; j--) {
            if (slots[j] == -1) {
                slots[j] = id;
                count++;
                profit += prof;
                break;
            }
        }
    }
    
    return {count, profit};
}

// Huffman Encoding
map<char, string> huffman_encoding(vector<pair<char, int>>& freq) {
    priority_queue<pair<int, string>, vector<pair<int, string>>, 
                   greater<pair<int, string>>> pq;
    
    map<char, string> codes;
    
    if (freq.size() == 1) {
        codes[freq[0].first] = "0";
        return codes;
    }
    
    // Initialize with characters
    for (auto& [ch, f] : freq) {
        string s(1, ch);
        pq.push({f, s});
    }
    
    while (pq.size() > 1) {
        auto [f1, s1] = pq.top(); pq.pop();
        auto [f2, s2] = pq.top(); pq.pop();
        
        // Add '0' prefix to s1, '1' prefix to s2
        string combined = s1 + s2;
        pq.push({f1 + f2, combined});
    }
    
    return codes;
}

// Minimum Railway Platforms needed
int minimum_platforms(vector<int>& arrival, vector<int>& departure) {
    sort(arrival.begin(), arrival.end());
    sort(departure.begin(), departure.end());
    
    int platforms_needed = 0;
    int max_platforms = 0;
    int i = 0, j = 0;
    int n = arrival.size();
    
    while (i < n) {
        if (arrival[i] <= departure[j]) {
            platforms_needed++;
            i++;
            max_platforms = max(max_platforms, platforms_needed);
        } else {
            platforms_needed--;
            j++;
        }
    }
    
    return max_platforms;
}

// ============= ADVANCED DP PATTERNS =============

// Matrix Chain Multiplication
int matrix_chain_multiplication(vector<int>& dims) {
    int n = dims.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i] * dims[k+1] * dims[j+1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n-1];
}

// Rod Cutting - Maximum profit
int rod_cutting(vector<int>& prices, int n) {
    vector<int> dp(n + 1, 0);
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i && j <= prices.size(); j++) {
            dp[i] = max(dp[i], prices[j-1] + dp[i-j]);
        }
    }
    
    return dp[n];
}

// Egg Drop Problem
int egg_drop(int eggs, int floors) {
    vector<vector<int>> dp(eggs + 1, vector<int>(floors + 1, INT_MAX));
    
    // Base cases
    for (int i = 0; i <= eggs; i++) {
        dp[i][0] = 0;
        dp[i][1] = 1;
    }
    
    for (int j = 0; j <= floors; j++) {
        dp[1][j] = j;
    }
    
    for (int i = 2; i <= eggs; i++) {
        for (int j = 2; j <= floors; j++) {
            for (int k = 1; k <= j; k++) {
                int worst = 1 + max(dp[i-1][k-1], dp[i][j-k]);
                dp[i][j] = min(dp[i][j], worst);
            }
        }
    }
    
    return dp[eggs][floors];
}

// ============= QUICK CP TEMPLATES =============

// Quick 0/1 Knapsack
int ks01(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++)
        for (int j = W; j >= w[i]; j--)
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    return dp[W];
}

// Quick Unbounded Knapsack
int ks_unb(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++)
        for (int j = w[i]; j <= W; j++)
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    return dp[W];
}

// Quick LCS
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
    
    cout << "=== 0/1 Knapsack ===" << endl;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};
    int capacity = 8;
    cout << "Max value: " << knapsack_01_1d(weights, values, capacity) << endl;
    
    auto [max_val, items] = knapsack_01_with_items(weights, values, capacity);
    cout << "Selected items: ";
    for (int i : items) cout << i << " ";
    cout << ", Value: " << max_val << endl;
    
    cout << "\n=== Unbounded Knapsack ===" << endl;
    cout << "Max value: " << knapsack_unbounded(weights, values, capacity) << endl;
    
    cout << "\n=== Subset Sum ===" << endl;
    vector<int> nums = {3, 34, 4, 12, 5, 2};
    int target = 9;
    cout << "Subset with sum " << target << " exists: " 
         << (subset_sum(nums, target) ? "Yes" : "No") << endl;
    cout << "Number of subsets: " << count_subset_sum(nums, target) << endl;
    
    cout << "\n=== Coin Change ===" << endl;
    vector<int> coins = {1, 2, 5};
    int amount = 11;
    cout << "Minimum coins: " << coin_change(coins, amount) << endl;
    cout << "Number of ways: " << coin_change_ways(coins, amount) << endl;
    
    cout << "\n=== LCS ===" << endl;
    string text1 = "abcde", text2 = "ace";
    cout << "LCS length: " << longest_common_subsequence(text1, text2) << endl;
    
    cout << "\n=== LIS ===" << endl;
    vector<int> nums2 = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "LIS length: " << longest_increasing_subsequence(nums2) << endl;
    
    cout << "\n=== Activity Selection ===" << endl;
    vector<int> start = {1, 3, 0, 5, 8, 5};
    vector<int> end = {2, 4, 6, 7, 9, 9};
    auto selected = activity_selection(start, end);
    cout << "Selected activities: ";
    for (int i : selected) cout << i << " ";
    cout << endl;
    
    cout << "\n=== Job Sequencing ===" << endl;
    vector<tuple<int, int, int>> jobs = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};
    auto [count, profit] = job_sequencing(jobs);
    cout << "Jobs: " << count << ", Total profit: " << profit << endl;
    
    return 0;
}


"""