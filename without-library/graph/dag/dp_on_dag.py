import sys
sys.setrecursionlimit(200000)

class DPDAG:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.memo = [-1] * (n + 1)
        
    def longest_path(self, u):
        if self.memo[u] != -1:
            return self.memo[u]
            
        mx = 0
        for v in self.adj[u]:
            val = 1 + self.longest_path(v)
            if val > mx: mx = val
            
        self.memo[u] = mx
        return mx

if __name__ == "__main__":
    pass
