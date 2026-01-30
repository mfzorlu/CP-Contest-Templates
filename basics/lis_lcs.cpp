/*
 * LIS (Longest Increasing Subsequence) and LCS (Longest Common Subsequence)
 * Dynamic Programming Templates for Competitive Programming
 * Complete collection with all variations
 */

#include <bits/stdc++.h>

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
