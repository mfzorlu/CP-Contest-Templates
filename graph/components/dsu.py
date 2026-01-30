import sys

# Fast I/O
input = sys.stdin.readline

class DSU:
    """
    Disjoint Set Union (DSU) / Union-Find with Path Compression and Union by Size
    Time Complexity: O(alpha(N)) which is nearly constant
    """
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
        self.num_sets = n
    
    def find(self, i):
        """Find representative of the set containing i with path compression"""
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        """Unions the sets containing i and j"""
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by size: attach smaller tree to larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_sets -= 1
            return True
        return False
    
    def get_size(self, i):
        return self.size[self.find(i)]

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        dsu = DSU(n)
        for _ in range(m):
            u, v = map(int, input().split())
            if dsu.union(u, v):
                print(f"Union {u} {v}: Merged")
            else:
                print(f"Union {u} {v}: Already in same set")
                
        print(f"Number of connected components: {dsu.num_sets}")
            
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
