import sys

# Fast I/O
input = sys.stdin.readline
sys.setrecursionlimit(200000)

class DFS:
    """
    DFS Template for CP
    Supports:
    - Recursive DFS
    - Iterative DFS
    - Entry/Exit times
    - Connected Components
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.visited = [False] * (n + 1)
        self.entry = [-1] * (n + 1)
        self.exit = [-1] * (n + 1)
        self.timer = 0
    
    def dfs_recursive(self, u):
        """Standard recursive DFS with entry/exit times"""
        self.visited[u] = True
        self.entry[u] = self.timer
        self.timer += 1
        
        for v in self.adj[u]:
            if not self.visited[v]:
                self.dfs_recursive(v)
        
        self.exit[u] = self.timer
        self.timer += 1
    
    def dfs_iterative(self, start_node):
        """Iterative DFS using stack (avoids recursion limit)"""
        stack = [start_node]
        
        while stack:
            u = stack.pop()
            if not self.visited[u]:
                self.visited[u] = True
                # Process node u
                
                # Add neighbors to stack
                for v in reversed(self.adj[u]):
                    if not self.visited[v]:
                        stack.append(v)
    
    def solve(self):
        """Find connected components"""
        components = 0
        for i in range(1, self.n + 1):
            if not self.visited[i]:
                components += 1
                self.dfs_recursive(i)
        return components

def solve():
    # Example input: N nodes, M edges
    # 5 4
    # 1 2
    # 2 3
    # 1 4
    # 4 5
    
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            adj[v].append(u)  # Undirected
            
        dfs_solver = DFS(n, adj)
        cc = dfs_solver.solve()
        print(f"Connected Components: {cc}")
        
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
