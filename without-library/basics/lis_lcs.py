import sys

# LIS / LCS without libraries

def lis(arr):
    if not arr: return 0
    n = len(arr)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    
    # Manual max
    mx = 0
    for x in dp:
        if x > mx: mx = x
    return mx

def lcs(s1, s2):
    n = len(s1)
    m = len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                v1 = dp[i-1][j]
                v2 = dp[i][j-1]
                dp[i][j] = v1 if v1 > v2 else v2
                
    return dp[n][m]

if __name__ == "__main__":
    pass
