"""
Coin Problems - Dynamic Programming Templates for Competitive Programming
All variations of coin/change problems with optimal solutions
"""

from typing import List, Tuple
import sys

# ============= PROBLEM 1: MINIMUM COINS TO MAKE AMOUNT =============

def coin_change_min(coins: List[int], amount: int) -> int:
    """
    Minimum number of coins to make exact amount
    Returns -1 if impossible
    Time: O(amount * len(coins)), Space: O(amount)
    
    Example: coins=[1,2,5], amount=11 → 3 (5+5+1)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_min_with_coins(coins: List[int], amount: int) -> Tuple[int, List[int]]:
    """
    Minimum coins with actual coin selection
    Returns: (min_count, list_of_coins_used)
    """
    dp = [float('inf')] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                if dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1
                    parent[i] = coin
    
    if dp[amount] == float('inf'):
        return -1, []
    
    # Reconstruct solution
    result = []
    curr = amount
    while curr > 0:
        coin_used = parent[curr]
        result.append(coin_used)
        curr -= coin_used
    
    return dp[amount], result


# ============= PROBLEM 2: NUMBER OF WAYS TO MAKE AMOUNT =============

def coin_change_ways(coins: List[int], amount: int) -> int:
    """
    Count number of ways to make amount (order doesn't matter)
    Example: coins=[1,2,5], amount=5 → 4 ways
    (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # IMPORTANT: Loop coins first to avoid counting permutations
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def coin_change_permutations(coins: List[int], amount: int) -> int:
    """
    Count number of ways where ORDER MATTERS (permutations)
    Example: coins=[1,2], amount=3 → 3 ways
    (1+1+1), (1+2), (2+1) - note (1+2) and (2+1) are different
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # IMPORTANT: Loop amount first to count permutations
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] += dp[i - coin]
    
    return dp[amount]


# ============= PROBLEM 3: COIN CHANGE WITH MODULO =============

def coin_change_ways_mod(coins: List[int], amount: int, mod: int = 10**9 + 7) -> int:
    """
    Number of ways to make amount (with modulo for large numbers)
    Common in competitive programming
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = (dp[i] + dp[i - coin]) % mod
    
    return dp[amount]


# ============= PROBLEM 4: COIN CHANGE WITH LIMITED COINS =============

def coin_change_limited(coins: List[int], counts: List[int], amount: int) -> int:
    """
    Minimum coins when each coin type has limited quantity
    coins: coin denominations
    counts: how many of each coin available
    
    Example: coins=[1,2,5], counts=[2,1,1], amount=7
    Can use at most 2 ones, 1 two, 1 five
    Time: O(amount * sum(counts)), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(len(coins)):
        coin = coins[i]
        count = counts[i]
        
        # Process each coin type with its limit
        for _ in range(count):
            for j in range(amount, coin - 1, -1):
                if dp[j - coin] != float('inf'):
                    dp[j] = min(dp[j], dp[j - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways_limited(coins: List[int], counts: List[int], amount: int) -> int:
    """
    Number of ways with limited coins
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for i in range(len(coins)):
        coin = coins[i]
        count = counts[i]
        
        for _ in range(count):
            for j in range(amount, coin - 1, -1):
                dp[j] += dp[j - coin]
    
    return dp[amount]


# ============= PROBLEM 5: MINIMUM/MAXIMUM COINS WITH EXACTLY K COINS =============

def coin_change_exact_k(coins: List[int], amount: int, k: int) -> bool:
    """
    Check if we can make amount using EXACTLY k coins
    Returns: True if possible, False otherwise
    Time: O(amount * k * len(coins)), Space: O(amount * k)
    """
    # dp[i][j] = can we make amount i using exactly j coins?
    dp = [[False] * (k + 1) for _ in range(amount + 1)]
    dp[0][0] = True
    
    for i in range(amount + 1):
        for j in range(k):
            if dp[i][j]:
                for coin in coins:
                    if i + coin <= amount:
                        dp[i + coin][j + 1] = True
    
    return dp[amount][k]


def coin_change_min_with_limit(coins: List[int], amount: int, max_coins: int) -> int:
    """
    Minimum coins to make amount, but using at most max_coins coins
    Returns -1 if impossible
    """
    # dp[i][j] = min coins to make amount i using at most j coins
    dp = [[float('inf')] * (max_coins + 1) for _ in range(amount + 1)]
    
    for j in range(max_coins + 1):
        dp[0][j] = 0
    
    for i in range(1, amount + 1):
        for j in range(1, max_coins + 1):
            for coin in coins:
                if coin <= i and dp[i - coin][j - 1] != float('inf'):
                    dp[i][j] = min(dp[i][j], dp[i - coin][j - 1] + 1)
    
    result = dp[amount][max_coins]
    return result if result != float('inf') else -1


# ============= PROBLEM 6: COIN CHANGE WITH DENOMINATIONS GENERATION =============

def coin_change_all_denominations(amount: int) -> int:
    """
    Minimum coins assuming we have all denominations [1,2,3,...,amount]
    This is always possible and equals the number of 1's needed
    But we can optimize using greedy approach
    """
    # With denominations [1,2,3,...,n], greedy works optimally
    # But for DP practice:
    coins = list(range(1, amount + 1))
    return coin_change_min(coins, amount)


# ============= PROBLEM 7: COIN CHANGE WITH RANGE =============

def coin_change_range(coins: List[int], min_amount: int, max_amount: int) -> List[int]:
    """
    For each amount in [min_amount, max_amount], find minimum coins
    Returns: list of minimum coins for each amount
    Time: O(max_amount * len(coins)), Space: O(max_amount)
    """
    dp = [float('inf')] * (max_amount + 1)
    dp[0] = 0
    
    for i in range(1, max_amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    result = []
    for amount in range(min_amount, max_amount + 1):
        result.append(dp[amount] if dp[amount] != float('inf') else -1)
    
    return result


# ============= PROBLEM 8: COIN CHANGE WITH TARGET SUM VARIATIONS =============

def coin_change_at_least(coins: List[int], min_amount: int) -> int:
    """
    Minimum coins to make AT LEAST min_amount
    """
    # Try amounts from min_amount to some reasonable upper bound
    max_try = min_amount + max(coins) if coins else min_amount
    
    min_coins = float('inf')
    for target in range(min_amount, max_try + 1):
        result = coin_change_min(coins, target)
        if result != -1:
            min_coins = min(min_coins, result)
            break
    
    return min_coins if min_coins != float('inf') else -1


def coin_change_closest(coins: List[int], target: int) -> Tuple[int, int]:
    """
    Find the closest amount we can make to target
    Returns: (closest_amount, min_coins_for_that_amount)
    """
    max_check = target + max(coins) if coins else target
    dp = [float('inf')] * (max_check + 1)
    dp[0] = 0
    
    for i in range(1, max_check + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    # Find closest amount to target
    closest_amount = -1
    min_diff = float('inf')
    min_coins = float('inf')
    
    for i in range(max_check + 1):
        if dp[i] != float('inf'):
            diff = abs(i - target)
            if diff < min_diff or (diff == min_diff and dp[i] < min_coins):
                min_diff = diff
                closest_amount = i
                min_coins = dp[i]
    
    return closest_amount, min_coins


# ============= PROBLEM 9: COIN CHANGE WITH MULTIPLE TARGETS =============

def coin_change_multiple_targets(coins: List[int], targets: List[int]) -> List[int]:
    """
    Efficiently solve for multiple target amounts
    Precompute DP once, then answer queries
    """
    max_target = max(targets)
    dp = [float('inf')] * (max_target + 1)
    dp[0] = 0
    
    for i in range(1, max_target + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    results = []
    for target in targets:
        results.append(dp[target] if dp[target] != float('inf') else -1)
    
    return results


# ============= PROBLEM 10: MAXIMUM VALUE WITH COIN WEIGHTS =============

def coin_knapsack(coins: List[int], values: List[int], capacity: int) -> int:
    """
    Coins have both weight and value (like knapsack)
    Maximize value while weight <= capacity
    This is unbounded knapsack variant
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(coins)):
        weight = coins[i]
        value = values[i]
        for w in range(weight, capacity + 1):
            dp[w] = max(dp[w], dp[w - weight] + value)
    
    return dp[capacity]


# ============= CSES PROBLEM SET - COIN PROBLEMS =============

def cses_minimizing_coins(coins: List[int], amount: int) -> int:
    """
    CSES: Minimizing Coins
    Your task is to find the minimum number of coins to make sum x
    """
    return coin_change_min(coins, amount)


def cses_coin_combinations_1(coins: List[int], amount: int) -> int:
    """
    CSES: Coin Combinations I (order matters - permutations)
    Count the number of ways you can produce sum x using coins
    """
    MOD = 10**9 + 7
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = (dp[i] + dp[i - coin]) % MOD
    
    return dp[amount]


def cses_coin_combinations_2(coins: List[int], amount: int) -> int:
    """
    CSES: Coin Combinations II (order doesn't matter - combinations)
    Count the number of distinct ways you can produce sum x
    """
    MOD = 10**9 + 7
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = (dp[i] + dp[i - coin]) % MOD
    
    return dp[amount]


# ============= QUICK CP TEMPLATES =============

def min_coins(coins, amount):
    """Ultra-compact minimum coins template"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i: dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


def ways_combinations(coins, amount, mod=10**9+7):
    """Ultra-compact ways (combinations) template"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] = (dp[i] + dp[i-c]) % mod
    return dp[amount]


def ways_permutations(coins, amount, mod=10**9+7):
    """Ultra-compact ways (permutations) template"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i: dp[i] = (dp[i] + dp[i-c]) % mod
    return dp[amount]


# ============= TESTING AND EXAMPLES =============

if __name__ == "__main__":
    print("=" * 60)
    print("COIN CHANGE PROBLEM VARIATIONS")
    print("=" * 60)
    
    coins = [1, 2, 5]
    amount = 11
    
    print("\n1. MINIMUM COINS")
    print(f"Coins: {coins}, Amount: {amount}")
    min_count = coin_change_min(coins, amount)
    print(f"Minimum coins needed: {min_count}")
    
    min_count, coin_list = coin_change_min_with_coins(coins, amount)
    print(f"Coins used: {coin_list}")
    
    print("\n2. NUMBER OF WAYS (Combinations - order doesn't matter)")
    print(f"Coins: {coins}, Amount: 5")
    ways = coin_change_ways(coins, 5)
    print(f"Number of ways: {ways}")
    print("Ways: (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)")
    
    print("\n3. NUMBER OF WAYS (Permutations - order matters)")
    coins2 = [1, 2]
    print(f"Coins: {coins2}, Amount: 3")
    perms = coin_change_permutations(coins2, 3)
    print(f"Number of permutations: {perms}")
    print("Ways: (1+1+1), (1+2), (2+1)")
    
    print("\n4. LIMITED COINS")
    coins3 = [1, 2, 5]
    counts = [2, 1, 1]
    amount3 = 7
    print(f"Coins: {coins3}, Counts: {counts}, Amount: {amount3}")
    result = coin_change_limited(coins3, counts, amount3)
    print(f"Minimum coins (with limits): {result}")
    
    print("\n5. EXACT K COINS")
    print(f"Coins: {coins}, Amount: 5, K: 3")
    can_make = coin_change_exact_k(coins, 5, 3)
    print(f"Can make with exactly 3 coins: {can_make}")
    print("Example: 2+2+1 = 5 using 3 coins")
    
    print("\n6. MULTIPLE TARGETS")
    targets = [1, 5, 10, 15, 20]
    print(f"Coins: {coins}, Targets: {targets}")
    results = coin_change_multiple_targets(coins, targets)
    print(f"Minimum coins for each target: {results}")
    
    print("\n7. CSES PROBLEMS")
    print("CSES - Minimizing Coins:")
    print(f"  Result: {cses_minimizing_coins([1, 5, 7], 11)}")
    
    print("CSES - Coin Combinations I (permutations):")
    print(f"  Result: {cses_coin_combinations_1([1, 5, 7], 11)}")
    
    print("CSES - Coin Combinations II (combinations):")
    print(f"  Result: {cses_coin_combinations_2([1, 5, 7], 11)}")
    
    print("\n8. EDGE CASES")
    print("Empty coins: ", coin_change_min([], 5))
    print("Amount 0: ", coin_change_min([1, 2, 5], 0))
    print("Impossible: ", coin_change_min([2, 5], 3))
    print("Large denominations: ", coin_change_min([3, 7], 10))
    
    print("\n" + "=" * 60)
    print("KEY DIFFERENCES TO REMEMBER:")
    print("=" * 60)
    print("COMBINATIONS (order doesn't matter): Loop coins FIRST")
    print("  for coin in coins:")
    print("      for i in range(coin, amount+1):")
    print()
    print("PERMUTATIONS (order matters): Loop amount FIRST")
    print("  for i in range(1, amount+1):")
    print("      for coin in coins:")



"""
/*
 * Coin Problems - Dynamic Programming Templates for Competitive Programming
 * All variations of coin/change problems with optimal solutions
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstring>

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
    /*
     * CSES: Minimizing Coins
     * Find the minimum number of coins to make sum x
     */
    return coin_change_min(coins, amount);
}

ll cses_coin_combinations_1(vector<int>& coins, int amount) {
    /*
     * CSES: Coin Combinations I (order matters - permutations)
     * Count the number of ways you can produce sum x
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = (dp[i] + dp[i - coin]) % MOD;
            }
        }
    }
    
    return dp[amount];
}

ll cses_coin_combinations_2(vector<int>& coins, int amount) {
    /*
     * CSES: Coin Combinations II (order doesn't matter - combinations)
     * Count the number of distinct ways you can produce sum x
     */
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = (dp[i] + dp[i - coin]) % MOD;
        }
    }
    
    return dp[amount];
}

// ============= QUICK CP TEMPLATES =============

// Ultra-compact minimum coins
int min_coins(vector<int>& c, int n) {
    vector<int> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++)
        for (int x : c)
            if (x <= i && dp[i-x] != INF)
                dp[i] = min(dp[i], dp[i-x] + 1);
    return dp[n] == INF ? -1 : dp[n];
}

// Ultra-compact ways (combinations)
ll ways_comb(vector<int>& c, int n) {
    vector<ll> dp(n + 1, 0);
    dp[0] = 1;
    for (int x : c)
        for (int i = x; i <= n; i++)
            dp[i] = (dp[i] + dp[i-x]) % MOD;
    return dp[n];
}

// Ultra-compact ways (permutations)
ll ways_perm(vector<int>& c, int n) {
    vector<ll> dp(n + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++)
        for (int x : c)
            if (x <= i)
                dp[i] = (dp[i] + dp[i-x]) % MOD;
    return dp[n];
}

// ============= MAIN - TESTING =============

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cout << string(60, '=') << endl;
    cout << "COIN CHANGE PROBLEM VARIATIONS" << endl;
    cout << string(60, '=') << endl;
    
    vector<int> coins = {1, 2, 5};
    int amount = 11;
    
    cout << "\n1. MINIMUM COINS\n";
    cout << "Coins: [1, 2, 5], Amount: " << amount << endl;
    int min_count = coin_change_min(coins, amount);
    cout << "Minimum coins needed: " << min_count << endl;
    
    auto [count, coin_list] = coin_change_min_with_coins(coins, amount);
    cout << "Coins used: ";
    for (int c : coin_list) cout << c << " ";
    cout << endl;
    
    cout << "\n2. NUMBER OF WAYS (Combinations - order doesn't matter)\n";
    cout << "Coins: [1, 2, 5], Amount: 5\n";
    ll ways = coin_change_ways(coins, 5);
    cout << "Number of ways: " << ways << endl;
    cout << "Ways: (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)\n";
    
    cout << "\n3. NUMBER OF WAYS (Permutations - order matters)\n";
    vector<int> coins2 = {1, 2};
    cout << "Coins: [1, 2], Amount: 3\n";
    ll perms = coin_change_permutations(coins2, 3);
    cout << "Number of permutations: " << perms << endl;
    cout << "Ways: (1+1+1), (1+2), (2+1)\n";
    
    cout << "\n4. LIMITED COINS\n";
    vector<int> coins3 = {1, 2, 5};
    vector<int> counts = {2, 1, 1};
    int amount3 = 7;
    cout << "Coins: [1, 2, 5], Counts: [2, 1, 1], Amount: " << amount3 << endl;
    int result = coin_change_limited(coins3, counts, amount3);
    cout << "Minimum coins (with limits): " << result << endl;
    
    cout << "\n5. EXACT K COINS\n";
    cout << "Coins: [1, 2, 5], Amount: 5, K: 3\n";
    bool can_make = coin_change_exact_k(coins, 5, 3);
    cout << "Can make with exactly 3 coins: " << (can_make ? "Yes" : "No") << endl;
    cout << "Example: 2+2+1 = 5 using 3 coins\n";
    
    cout << "\n6. MULTIPLE TARGETS\n";
    vector<int> targets = {1, 5, 10, 15, 20};
    cout << "Coins: [1, 2, 5], Targets: [1, 5, 10, 15, 20]\n";
    vector<int> results = coin_change_multiple_targets(coins, targets);
    cout << "Minimum coins for each target: ";
    for (int r : results) cout << r << " ";
    cout << endl;
    
    cout << "\n7. CSES PROBLEMS\n";
    vector<int> cses_coins = {1, 5, 7};
    cout << "CSES - Minimizing Coins:\n";
    cout << "  Result: " << cses_minimizing_coins(cses_coins, 11) << endl;
    
    cout << "CSES - Coin Combinations I (permutations):\n";
    cout << "  Result: " << cses_coin_combinations_1(cses_coins, 11) << endl;
    
    cout << "CSES - Coin Combinations II (combinations):\n";
    cout << "  Result: " << cses_coin_combinations_2(cses_coins, 11) << endl;
    
    cout << "\n8. EDGE CASES\n";
    vector<int> empty_coins;
    cout << "Empty coins: " << coin_change_min(empty_coins, 5) << endl;
    cout << "Amount 0: " << coin_change_min(coins, 0) << endl;
    
    vector<int> coins4 = {2, 5};
    cout << "Impossible: " << coin_change_min(coins4, 3) << endl;
    
    vector<int> coins5 = {3, 7};
    cout << "Large denominations: " << coin_change_min(coins5, 10) << endl;
    
    cout << "\n" << string(60, '=') << endl;
    cout << "KEY DIFFERENCES TO REMEMBER:" << endl;
    cout << string(60, '=') << endl;
    cout << "COMBINATIONS (order doesn't matter): Loop coins FIRST\n";
    cout << "  for (int coin : coins)\n";
    cout << "      for (int i = coin; i <= amount; i++)\n\n";
    
    cout << "PERMUTATIONS (order matters): Loop amount FIRST\n";
    cout << "  for (int i = 1; i <= amount; i++)\n";
    cout << "      for (int coin : coins)\n";
    
    return 0;
}



"""