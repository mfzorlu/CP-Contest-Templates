import sys

class TopoSort:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.in_degree = [0] * (n + 1)
        
    def solve(self):
        for u in range(1, self.n + 1):
            for v in self.adj[u]:
                self.in_degree[v] += 1
                
        queue = []
        for i in range(1, self.n + 1):
            if self.in_degree[i] == 0:
                queue.append(i)
                
        head = 0
        topo = []
        
        while head < len(queue):
            u = queue[head]
            head += 1
            topo.append(u)
            
            for v in self.adj[u]:
                self.in_degree[v] -= 1
                if self.in_degree[v] == 0:
                    queue.append(v)
                    
        return topo if len(topo) == self.n else []

if __name__ == "__main__":
    pass
