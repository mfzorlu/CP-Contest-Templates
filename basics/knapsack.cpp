/*
 * Knapsack, DP, and Greedy Algorithms for Competitive Programming
 * Complete collection of optimization problems
 */

#include <bits/stdc++.h>

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
