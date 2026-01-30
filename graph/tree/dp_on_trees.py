import sys

# Fast I/O
input = sys.stdin.readline
sys.setrecursionlimit(200000)

class TreeDP:
    """
    DP on Trees Template
    Example: Tree Diameter, Maximum Independent Set
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.dp = [[0] * 2 for _ in range(n + 1)] # Adjust dimensions as needed
        self.parent = [-1] * (n + 1)
        self.max_dist = -1
        self.farthest_node = -1
    
    def dfs_diameter(self, u, p, d):
        """Find farthest node (for diameter)"""
        if d > self.max_dist:
            self.max_dist = d
            self.farthest_node = u
            
        for v in self.adj[u]:
            if v != p:
                self.dfs_diameter(v, u, d + 1)
    
    def get_diameter(self):
        """
        Returns diameter of the tree using double DFS
        """
        self.max_dist = -1
        self.dfs_diameter(1, -1, 0)
        
        start_node = self.farthest_node
        self.max_dist = -1
        self.dfs_diameter(start_node, -1, 0)
        
        return self.max_dist
    
    def dfs_dp(self, u, p):
        """
        Generic DP on Tree structure
        Example: Max Independent Set
        dp[u][0] = max set stats without u
        dp[u][1] = max set stats with u
        """
        self.dp[u][0] = 0
        self.dp[u][1] = 1 # Weight of u (1 if unweighted)
        
        for v in self.adj[u]:
            if v != p:
                self.dfs_dp(v, u)
                
                # If we don't take u, we can either take v or not
                self.dp[u][0] += max(self.dp[v][0], self.dp[v][1])
                
                # If we take u, we cannot take v
                self.dp[u][1] += self.dp[v][0]

    def solve_mis(self):
        self.dfs_dp(1, -1)
        return max(self.dp[1][0], self.dp[1][1])

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        # Tree has n-1 edges usually, but reading m for generality
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            adj[v].append(u)
            
        solver = TreeDP(n, adj)
        print(f"Tree Diameter: {solver.get_diameter()}")
        print(f"Max Independent Set Size: {solver.solve_mis()}")
            
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
