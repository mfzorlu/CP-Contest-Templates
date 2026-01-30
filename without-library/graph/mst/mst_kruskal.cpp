#include <iostream>
#include <vector>
#include <algorithm> // for sort

using namespace std;

struct Edge {
    int u, v, w;
    // Manual comparator for sort
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

struct DSU {
    vector<int> parent;
    DSU(int n) {
        parent.resize(n + 1);
        for(int i=0; i<=n; i++) parent[i] = i;
    }
    int find(int i) {
        if(parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if(root_i != root_j) {
            parent[root_j] = root_i;
            return true;
        }
        return false;
    }
};

int main() {
    int n, m;
    // ... input ...
    vector<Edge> edges;
    // ... push edges ...
    
    sort(edges.begin(), edges.end());
    
    DSU dsu(n);
    // ... kruskal logic ...
    
    return 0;
}
