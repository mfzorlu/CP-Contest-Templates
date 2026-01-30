import sys

# Bipartite Check (BFS)

class Bipartite:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.color = [-1] * (n + 1)
        
    def check(self, start):
        queue = [start]
        head = 0
        self.color[start] = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for v in self.adj[u]:
                if self.color[v] == -1:
                    self.color[v] = 1 - self.color[u]
                    queue.append(v)
                elif self.color[v] == self.color[u]:
                    return False
        return True

if __name__ == "__main__":
    pass
