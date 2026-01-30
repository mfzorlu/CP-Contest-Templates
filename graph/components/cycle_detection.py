import sys

# Fast I/O
input = sys.stdin.readline
sys.setrecursionlimit(200000)

class CycleDetection:
    """
    Cycle Detection Templates
    """
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.visited = [False] * (n + 1)
        self.recursion_stack = [False] * (n + 1)
        self.parent = [-1] * (n + 1)
        self.cycle_start = -1
        self.cycle_end = -1
    
    def find_cycle_undirected(self, u, p=-1):
        """
        Detect cycle in undirected graph using DFS
        Returns True if cycle found
        """
        self.visited[u] = True
        self.parent[u] = p
        
        for v in self.adj[u]:
            if v == p:
                continue
            if self.visited[v]:
                self.cycle_start = v
                self.cycle_end = u
                return True
            if self.find_cycle_undirected(v, u):
                return True
        return False

    def find_cycle_directed(self, u):
        """
        Detect cycle in directed graph using 3-color DFS approach
        States could be: 0 (unvisited), 1 (visiting), 2 (visited)
        Here simplified with visited and recursion_stack arrays
        """
        self.visited[u] = True
        self.recursion_stack[u] = True
        
        for v in self.adj[u]:
            if not self.visited[v]:
                self.parent[v] = u
                if self.find_cycle_directed(v):
                    return True
            elif self.recursion_stack[v]:
                self.cycle_start = v
                self.cycle_end = u
                return True
        
        self.recursion_stack[u] = False
        return False

    def get_cycle_path(self):
        """Reconstruct the cycle path"""
        if self.cycle_start == -1:
            return []
        
        cycle = []
        cycle.append(self.cycle_start)
        curr = self.cycle_end
        while curr != self.cycle_start and curr != -1:
            cycle.append(curr)
            curr = self.parent[curr]
        cycle.append(self.cycle_start)
        return cycle[::-1]

def solve():
    # Example usage
    try:
        line1 = input().split()
        if not line1: return
        n, m = map(int, line1)
        
        adj = [[] for _ in range(n + 1)]
        is_directed = False # Change as needed
        
        for _ in range(m):
            u, v = map(int, input().split())
            adj[u].append(v)
            if not is_directed:
                adj[v].append(u)
        
        detector = CycleDetection(n, adj)
        has_cycle = False
        
        for i in range(1, n + 1):
            if not detector.visited[i]:
                if is_directed:
                    if detector.find_cycle_directed(i):
                        has_cycle = True
                        break
                else:
                    if detector.find_cycle_undirected(i):
                        has_cycle = True
                        break
        
        if has_cycle:
            print("Cycle found")
            print("Path:", detector.get_cycle_path())
        else:
            print("No cycle")

    except ValueError:
        pass

if __name__ == "__main__":
    solve()
