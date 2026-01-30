import sys

# DFS without libraries
# Standard recursion is fine (built-in)

sys.setrecursionlimit(200000)

class DFS:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.visited = [False] * (n + 1)
        self.entry = [-1] * (n + 1)
        self.exit = [-1] * (n + 1)
        self.timer = 0
    
    def dfs(self, u):
        self.visited[u] = True
        self.entry[u] = self.timer
        self.timer += 1
        
        for v in self.adj[u]:
            if not self.visited[v]:
                self.dfs(v)
                
        self.exit[u] = self.timer
        self.timer += 1

def solve():
    # Input reading example not needed for template
    pass
