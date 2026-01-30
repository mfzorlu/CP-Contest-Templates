import sys
from collections import deque

# Fast I/O
input = sys.stdin.readline

class BFS:
    """
    BFS Template for CP
    Supports:
    - Shortest Path (Unweighted)
    - Level Traversal
    - 0-1 BFS (commented)
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.dist = [-1] * (n + 1)
        self.parent = [-1] * (n + 1)
    
    def bfs(self, start_node):
        """Standard BFS"""
        q = deque([start_node])
        self.dist[start_node] = 0
        self.parent[start_node] = -1
        
        while q:
            u = q.popleft()
            
            for v in self.adj[u]:
                if self.dist[v] == -1:
                    self.dist[v] = self.dist[u] + 1
                    self.parent[v] = u
                    q.append(v)
    
    def get_path(self, end_node):
        """Reconstruct path from start_node to end_node"""
        if self.dist[end_node] == -1:
            return []
        
        path = []
        curr = end_node
        while curr != -1:
            path.append(curr)
            curr = self.parent[curr]
        
        return path[::-1]

def solve():
    # Example input
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            adj[v].append(u)
            
        bfs_solver = BFS(n, adj)
        bfs_solver.bfs(1) # Start from node 1
        
        print(f"Distances from 1: {bfs_solver.dist[1:]}")
        
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
