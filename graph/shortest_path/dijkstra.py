import sys
from heapq import heappush, heappop

# Fast I/O
input = sys.stdin.readline

class Dijkstra:
    """
    Dijkstra's Algorithm using Priority Queue
    Finds shortest paths from source to all other nodes
    Graph must have non-negative edge weights
    Time Complexity: O(E log V)
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.INF = float('inf')
        self.dist = [self.INF] * (n + 1)
        self.parent = [-1] * (n + 1)
    
    def solve(self, start_node):
        self.dist[start_node] = 0
        pq = [(0, start_node)] # (distance, node)
        
        while pq:
            d, u = heappop(pq)
            
            # Optimization: If we found a shorter path to u before, skip
            if d > self.dist[u]:
                continue
            
            for v, weight in self.adj[u]:
                if self.dist[u] + weight < self.dist[v]:
                    self.dist[v] = self.dist[u] + weight
                    self.parent[v] = u
                    heappush(pq, (self.dist[v], v))
                    
    def get_path(self, end_node):
        if self.dist[end_node] == self.INF:
            return []
        
        path = []
        curr = end_node
        while curr != -1:
            path.append(curr)
            curr = self.parent[curr]
        return path[::-1]

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            # u, v, w
            u, v, w = map(int, input().split())
            adj[u].append((v, w))
            adj[v].append((u, w)) # Undirected
            
        dijkstra = Dijkstra(n, adj)
        dijkstra.solve(1)
        
        for i in range(1, n + 1):
            d = dijkstra.dist[i]
            if d == float('inf'):
                print(f"Node {i}: Unreachable")
            else:
                print(f"Node {i}: {d}")
                
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
