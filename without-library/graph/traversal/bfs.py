import sys

# BFS without deque/collections
# Using list with pointer for O(1) avg pop

class BFS:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.dist = [-1] * (n + 1)
        self.parent = [-1] * (n + 1)
        
    def bfs(self, start):
        # Manual queue using list
        queue = [start]
        head = 0 # Pointer to front
        
        self.dist[start] = 0
        self.parent[start] = -1
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for v in self.adj[u]:
                if self.dist[v] == -1:
                    self.dist[v] = self.dist[u] + 1
                    self.parent[v] = u
                    queue.append(v)

if __name__ == "__main__":
    pass
