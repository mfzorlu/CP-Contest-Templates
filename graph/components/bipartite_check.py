import sys
from collections import deque

# Fast I/O
input = sys.stdin.readline
sys.setrecursionlimit(200000)

class BipartiteCheck:
    """
    Check if a graph is Bipartite (2-colorable)
    Time Complexity: O(V + E)
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.color = [-1] * (n + 1) # -1: uncolored, 0: color A, 1: color B
        self.is_bipartite = True

    def check_bfs(self, start_node):
        """Returns False if not bipartite"""
        q = deque([start_node])
        self.color[start_node] = 0
        
        while q:
            u = q.popleft()
            
            for v in self.adj[u]:
                if self.color[v] == -1:
                    self.color[v] = 1 - self.color[u]
                    q.append(v)
                elif self.color[v] == self.color[u]:
                    self.is_bipartite = False
                    return False
        return True

    def solve(self):
        for i in range(1, self.n + 1):
            if self.color[i] == -1:
                if not self.check_bfs(i):
                    return False
        return True

def solve():
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            adj[v].append(u)
            
        checker = BipartiteCheck(n, adj)
        if checker.solve():
            print("YES")
            # Print coloring
            # print(checker.color[1:])
        else:
            print("NO")
            
    except ValueError:
        pass

if __name__ == "__main__":
    solve()
