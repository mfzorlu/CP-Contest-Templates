/*
 * Coin Problems - Dynamic Programming Templates for Competitive Programming
 * All variations of coin/change problems with optimal solutions
 */

#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

const int MOD = 1e9 + 7;
const int INF = 1e9;

// ============= PROBLEM 1: MINIMUM COINS TO MAKE AMOUNT =============

int coin_change_min(vector<int>& coins, int amount) {
    /*
     * Minimum number of coins to make exact amount
     * Returns -1 if impossible
     * Time: O(amount * n), Space: O(amount)
     * Example: coins=[1,2,5], amount=11 → 3 (5+5+1)
     */
    vector<int> dp(amount + 1, INF);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INF) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] == INF ? -1 : dp[amount];
}

pair<int, vector<int>> coin_change_min_with_coins(vector<int>& coins, int amount) {
    /*
     * Minimum coins with actual coin selection
     * Returns: {min_count, vector_of_coins_used}
     */
    vector<int> dp(amount + 1, INF);
    vector<int> parent(amount + 1, -1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INF) {
                if (dp[i - coin] + 1 < dp[i]) {
                    dp[i] = dp[i - coin] + 1;
                    parent[i] = coin;
                }
            }
        }
    }
    
    if (dp[amount] == INF) {
        return {-1, {}};
    }
    
    // Reconstruct solution
    vector<int> result;
    int curr = amount;
    while (curr > 0) {
        int coin_used = parent[curr];
        result.push_back(coin_used);
        curr -= coin_used;
    }
    
    return {dp[amount], result};
}

// ============= PROBLEM 2: NUMBER OF WAYS TO MAKE AMOUNT =============

ll coin_change_ways(vector<int>& coins, int amount) {
    /*
     * Count number of ways to make amount (order doesn't matter - COMBINATIONS)
     * Example: coins=[1,2,5], amount=5 → 4 ways
     * (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)
     * Time: O(amount * n), Space: O(amount)
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    // IMPORTANT: Loop coins first to avoid counting permutations
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    
    return dp[amount];
}

ll coin_change_permutations(vector<int>& coins, int amount) {
    /*
     * Count number of ways where ORDER MATTERS (permutations)
     * Example: coins=[1,2], amount=3 → 3 ways
     * (1+1+1), (1+2), (2+1) - note (1+2) and (2+1) are different
     * Time: O(amount * n), Space: O(amount)
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    // IMPORTANT: Loop amount first to count permutations
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] += dp[i - coin];
            }
        }
    }
    
    return dp[amount];
}

// ============= PROBLEM 3: COIN CHANGE WITH MODULO =============

ll coin_change_ways_mod(vector<int>& coins, int amount, ll mod = MOD) {
    /*
     * Number of ways to make amount (with modulo for large numbers)
     * Common in competitive programming
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = (dp[i] + dp[i - coin]) % mod;
        }
    }
    
    return dp[amount];
}

ll coin_change_permutations_mod(vector<int>& coins, int amount, ll mod = MOD) {
    /*
     * Number of permutations with modulo
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = (dp[i] + dp[i - coin]) % mod;
            }
        }
    }
    
    return dp[amount];
}

// ============= PROBLEM 4: COIN CHANGE WITH LIMITED COINS =============

int coin_change_limited(vector<int>& coins, vector<int>& counts, int amount) {
    /*
     * Minimum coins when each coin type has limited quantity
     * coins: coin denominations
     * counts: how many of each coin available
     * Time: O(amount * sum(counts)), Space: O(amount)
     */
    vector<int> dp(amount + 1, INF);
    dp[0] = 0;
    
    for (int i = 0; i < coins.size(); i++) {
        int coin = coins[i];
        int count = counts[i];
        
        // Process each coin type with its limit
        for (int k = 0; k < count; k++) {
            for (int j = amount; j >= coin; j--) {
                if (dp[j - coin] != INF) {
                    dp[j] = min(dp[j], dp[j - coin] + 1);
                }
            }
        }
    }
    
    return dp[amount] == INF ? -1 : dp[amount];
}

ll coin_change_ways_limited(vector<int>& coins, vector<int>& counts, int amount) {
    /*
     * Number of ways with limited coins
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int i = 0; i < coins.size(); i++) {
        int coin = coins[i];
        int count = counts[i];
        
        for (int k = 0; k < count; k++) {
            for (int j = amount; j >= coin; j--) {
                dp[j] += dp[j - coin];
            }
        }
    }
    
    return dp[amount];
}

// ============= PROBLEM 5: MINIMUM/MAXIMUM COINS WITH EXACTLY K COINS =============

bool coin_change_exact_k(vector<int>& coins, int amount, int k) {
    /*
     * Check if we can make amount using EXACTLY k coins
     * Time: O(amount * k * n), Space: O(amount * k)
     */
    vector<vector<bool>> dp(amount + 1, vector<bool>(k + 1, false));
    dp[0][0] = true;
    
    for (int i = 0; i <= amount; i++) {
        for (int j = 0; j < k; j++) {
            if (dp[i][j]) {
                for (int coin : coins) {
                    if (i + coin <= amount) {
                        dp[i + coin][j + 1] = true;
                    }
                }
            }
        }
    }
    
    return dp[amount][k];
}

int coin_change_min_with_limit(vector<int>& coins, int amount, int max_coins) {
    /*
     * Minimum coins to make amount, using at most max_coins coins
     * Returns -1 if impossible
     */
    vector<vector<int>> dp(amount + 1, vector<int>(max_coins + 1, INF));
    
    for (int j = 0; j <= max_coins; j++) {
        dp[0][j] = 0;
    }
    
    for (int i = 1; i <= amount; i++) {
        for (int j = 1; j <= max_coins; j++) {
            for (int coin : coins) {
                if (coin <= i && dp[i - coin][j - 1] != INF) {
                    dp[i][j] = min(dp[i][j], dp[i - coin][j - 1] + 1);
                }
            }
        }
    }
    
    return dp[amount][max_coins] == INF ? -1 : dp[amount][max_coins];
}

// ============= PROBLEM 6: COIN CHANGE WITH RANGE =============

vector<int> coin_change_range(vector<int>& coins, int min_amount, int max_amount) {
    /*
     * For each amount in [min_amount, max_amount], find minimum coins
     * Returns: vector of minimum coins for each amount
     */
    vector<int> dp(max_amount + 1, INF);
    dp[0] = 0;
    
    for (int i = 1; i <= max_amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INF) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    vector<int> result;
    for (int amount = min_amount; amount <= max_amount; amount++) {
        result.push_back(dp[amount] == INF ? -1 : dp[amount]);
    }
    
    return result;
}

// ============= PROBLEM 7: COIN CHANGE WITH MULTIPLE TARGETS =============

vector<int> coin_change_multiple_targets(vector<int>& coins, vector<int>& targets) {
    /*
     * Efficiently solve for multiple target amounts
     * Precompute DP once, then answer queries
     */
    int max_target = *max_element(targets.begin(), targets.end());
    vector<int> dp(max_target + 1, INF);
    dp[0] = 0;
    
    for (int i = 1; i <= max_target; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INF) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    vector<int> results;
    for (int target : targets) {
        results.push_back(dp[target] == INF ? -1 : dp[target]);
    }
    
    return results;
}

// ============= CSES PROBLEM SET - COIN PROBLEMS =============

int cses_minimizing_coins(vector<int>& coins, int amount) {
    return coin_change_min(coins, amount);
}
