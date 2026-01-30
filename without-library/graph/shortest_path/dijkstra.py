import sys

# Dijkstra with Manual Min-Heap

class MinHeap:
    def __init__(self):
        self.heap = []
        
    def push(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)
        
    def pop(self):
        if not self.heap:
            return None
        min_item = self.heap[0]
        last_item = self.heap.pop()
        if self.heap:
            self.heap[0] = last_item
            self._sift_down(0)
        return min_item
        
    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        if idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self._sift_up(parent)
            
    def _sift_down(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
            
        if smallest != idx:
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            self._sift_down(smallest)

    def is_empty(self):
        return len(self.heap) == 0

class Dijkstra:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.INF = 10**18
        self.dist = [self.INF] * (n + 1)
        
    def solve(self, start):
        self.dist[start] = 0
        pq = MinHeap()
        pq.push((0, start))
        
        while not pq.is_empty():
            d, u = pq.pop()
            
            if d > self.dist[u]:
                continue
                
            for v, w in self.adj[u]:
                if self.dist[u] + w < self.dist[v]:
                    self.dist[v] = self.dist[u] + w
                    pq.push((self.dist[v], v))

if __name__ == "__main__":
    pass
