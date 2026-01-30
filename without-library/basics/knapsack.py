import sys

# Knapsack Problem without libraries

def knapsack(weights, values, capacity):
    n = len(weights)
    # Manual 2D array creation
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't take item
            dp[i][w] = dp[i-1][w]
            
            # Option 2: Take item if it fits
            if weights[i-1] <= w:
                val = dp[i-1][w - weights[i-1]] + values[i-1]
                if val > dp[i][w]:
                    dp[i][w] = val
                    
    return dp[n][capacity]

if __name__ == "__main__":
    pass
