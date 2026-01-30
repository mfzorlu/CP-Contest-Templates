#include <bits/stdc++.h>
using namespace std;

/*
 * Kruskal's Algorithm for Minimum Spanning Tree (MST)
 * Time Complexity: O(E log E)
 */

struct Edge {
    int u, v, w;
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

struct DSU {
    vector<int> parent;
    vector<int> size;
    
    DSU(int n) {
        parent.resize(n + 1);
        size.assign(n + 1, 1);
        iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (size[root_i] < size[root_j]) swap(root_i, root_j);
            parent[root_j] = root_i;
            size[root_i] += size[root_j];
            return true;
        }
        return false;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    vector<Edge> edges;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        edges.push_back({u, v, w});
    }
    
    sort(edges.begin(), edges.end());
    
    DSU dsu(n);
    long long mst_weight = 0;
    int edges_count = 0;
    vector<Edge> mst_edges;
    
    for (const auto& edge : edges) {
        if (dsu.unite(edge.u, edge.v)) {
            mst_weight += edge.w;
            mst_edges.push_back(edge);
            edges_count++;
        }
    }
    
    if (edges_count == n - 1) {
        cout << "MST Weight: " << mst_weight << "\n";
    } else {
        cout << "IMPOSSIBLE\n";
    }
    
    return 0;
}
