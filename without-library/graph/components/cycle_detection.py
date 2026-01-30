import sys

sys.setrecursionlimit(200000)

class CycleDetection:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.visited = [False] * (n + 1)
        # For directed
        self.rec_stack = [False] * (n + 1)
        
    def find_undirected(self, u, p=-1):
        self.visited[u] = True
        for v in self.adj[u]:
            if v == p: continue
            if self.visited[v]: return True
            if self.find_undirected(v, u): return True
        return False
        
    def find_directed(self, u):
        self.visited[u] = True
        self.rec_stack[u] = True
        for v in self.adj[u]:
            if not self.visited[v]:
                if self.find_directed(v): return True
            elif self.rec_stack[v]:
                return True
        self.rec_stack[u] = False
        return False

if __name__ == "__main__":
    pass
