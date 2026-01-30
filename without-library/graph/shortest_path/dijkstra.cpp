#include <iostream>
#include <vector>

using namespace std;

typedef long long ll;
const ll INF = 1e18;

// Manual Min-Heap
struct MinHeap {
    vector<pair<ll, int>> heap;
    
    void push(pair<ll, int> val) {
        heap.push_back(val);
        sift_up(heap.size() - 1);
    }
    
    pair<ll, int> pop() {
        pair<ll, int> res = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) sift_down(0);
        return res;
    }
    
    bool empty() {
        return heap.empty();
    }
    
    void sift_up(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[idx] < heap[parent]) {
                pair<ll, int> temp = heap[idx];
                heap[idx] = heap[parent];
                heap[parent] = temp;
                idx = parent;
            } else {
                break;
            }
        }
    }
    
    void sift_down(int idx) {
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            
            if (left < heap.size() && heap[left] < heap[smallest])
                smallest = left;
            if (right < heap.size() && heap[right] < heap[smallest])
                smallest = right;
                
            if (smallest != idx) {
                pair<ll, int> temp = heap[idx];
                heap[idx] = heap[smallest];
                heap[smallest] = temp;
                idx = smallest;
            } else {
                break;
            }
        }
    }
};

struct Edge {
    int to;
    int weight;
};

vector<Edge> adj[200005];
ll dist[200005];

void dijkstra(int start, int n) {
    for(int i=0; i<=n; i++) dist[i] = INF;
    
    MinHeap pq;
    dist[start] = 0;
    pq.push({0, start});
    
    while(!pq.empty()) {
        pair<ll, int> top = pq.pop();
        ll d = top.first;
        int u = top.second;
        
        if (d > dist[u]) continue;
        
        for(size_t i=0; i<adj[u].size(); i++) {
            int v = adj[u][i].to;
            int w = adj[u][i].weight;
            
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
}

int main() {
    return 0;
}
