import sys

# Fast I/O
input = sys.stdin.readline

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

class KruskalMVP:
    """
    Kruskal's Algorithm for Minimum Spanning Tree (MST)
    Time Complexity: O(E log E) or O(E log V)
    """
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges # List of (u, v, w)
    
    def solve(self):
        # Sort edges by weight
        self.edges.sort(key=lambda x: x[2])
        
        dsu = DSU(self.n)
        mst_weight = 0
        mst_edges = []
        edges_count = 0
        
        for u, v, w in self.edges:
            if dsu.union(u, v):
                mst_weight += w
                mst_edges.append((u, v, w))
                edges_count += 1
        
        if edges_count == self.n - 1:
            return mst_weight, mst_edges
        else:
            return -1, [] # Graph not connected

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        edges = []
        for _ in range(m):
            u, v, w = map(int, input().split())
            edges.append((u, v, w))
            
        algo = KruskalMVP(n, edges)
        weight, mst = algo.solve()
        
        if weight != -1:
            print(f"MST Weight: {weight}")
            # for u, v, w in mst:
            #     print(f"{u} - {v}: {w}")
        else:
            print("IMPOSSIBLE")
            
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
