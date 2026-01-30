import sys
from collections import deque

# Fast I/O
input = sys.stdin.readline

class TopologicalSort:
    """
    Topological Sort (Kahn's Algorithm)
    Works on DAG (Directed Acyclic Graph)
    Time Complexity: O(V + E)
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.in_degree = [0] * (n + 1)
        self.compute_indegree()
        
    def compute_indegree(self):
        for u in range(1, self.n + 1):
            for v in self.adj[u]:
                self.in_degree[v] += 1
                
    def solve_kahn(self):
        """Returns topological ordering or [] if cycle detected"""
        q = deque()
        for i in range(1, self.n + 1):
            if self.in_degree[i] == 0:
                q.append(i)
        
        topo_order = []
        while q:
            # Use heapq here if you need lexicographically smallest sort
            u = q.popleft()
            topo_order.append(u)
            
            for v in self.adj[u]:
                self.in_degree[v] -= 1
                if self.in_degree[v] == 0:
                    q.append(v)
                    
        if len(topo_order) < self.n:
            return [] # Cycle detected
            
        return topo_order

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            # Directed graph
        
        topo = TopologicalSort(n, adj)
        result = topo.solve_kahn()
        
        if not result:
            print("IMPOSSIBLE") # Cycle
        else:
            print(*result)
            
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
