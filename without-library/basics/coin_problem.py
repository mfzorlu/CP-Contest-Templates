import sys

# Coin Problem without libraries

def solve_min_coins(coins, amount):
    # Initialize DP array
    # Using specific large number instead of float('inf') for manual feel
    INF = 10**9
    dp = [INF] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                if dp[i - coin] != INF:
                    if dp[i - coin] + 1 < dp[i]:
                        dp[i] = dp[i - coin] + 1
                        
    return -1 if dp[amount] == INF else dp[amount]

if __name__ == "__main__":
    pass
