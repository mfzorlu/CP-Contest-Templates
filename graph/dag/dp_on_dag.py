import sys
sys.setrecursionlimit(200000)

# Fast I/O
input = sys.stdin.readline

class DPDAG:
    """
    DP on DAG (Directed Acyclic Graph)
    Example: Longest Path
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.memo = [-1] * (n + 1)
        self.parent = [-1] * (n + 1) # For path reconstruction
    
    def longest_path(self, u):
        """Returns length of longest path starting from u"""
        if self.memo[u] != -1:
            return self.memo[u]
        
        max_len = 0
        for v in self.adj[u]:
            val = 1 + self.longest_path(v)
            if val > max_len:
                max_len = val
                self.parent[u] = v
        
        self.memo[u] = max_len
        return max_len

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            
        solver = DPDAG(n, adj)
        
        global_max = 0
        global_start = -1
        
        for i in range(1, n + 1):
            val = solver.longest_path(i)
            if val > global_max:
                global_max = val
                global_start = i
                
        print(f"Longest Path Length: {global_max}")
        
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
