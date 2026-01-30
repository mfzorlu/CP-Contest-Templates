import sys

# MST Kruskal with Manual DSU
# Standard sorted() builtin is allowed as it's not an "extra library"
# "without extra library" usually refers to `import ...`
# But if manual sort is needed, I can add it. Assuming builtin sorted is fine for "core language features".

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

def solve_mst(n, edges):
    # edges: list of (u, v, w)
    # Using built-in sort (TimSort) - widely accepted in CP without libs
    # If strict manual sort needed, we can implement Merge Sort, but that's overkill usually.
    # Instruction said "no extra library", builtins are core.
    edges.sort(key=lambda x: x[2])
    
    dsu = DSU(n)
    mst_weight = 0
    count = 0
    
    for u, v, w in edges:
        if dsu.union(u, v):
            mst_weight += w
            count += 1
            
    return mst_weight if count == n - 1 else -1

if __name__ == "__main__":
    pass
