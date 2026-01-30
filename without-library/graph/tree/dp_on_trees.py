import sys
sys.setrecursionlimit(200000)

class TreeDP:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.dp = [[0, 0] for _ in range(n + 1)]
        
    def dfs(self, u, p):
        self.dp[u][0] = 0
        self.dp[u][1] = 1
        
        for v in self.adj[u]:
            if v != p:
                self.dfs(v, u)
                self.dp[u][0] += max(self.dp[v][0], self.dp[v][1])
                self.dp[u][1] += self.dp[v][0]

if __name__ == "__main__":
    pass
