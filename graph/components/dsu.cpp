#include <bits/stdc++.h>
using namespace std;

/*
 * Disjoint Set Union (DSU) / Union-Find
 * Optimizations: Path Compression + Union by Size
 * Time Complexity: O(alpha(N)) ≈ O(1)
 */

struct DSU {
    vector<int> parent;
    vector<int> size;
    int num_sets;
    
    DSU(int n) {
        parent.resize(n + 1);
        size.assign(n + 1, 1);
        num_sets = n;
        // Initialize parent to self
        iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]); // Path compression
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        
        if (root_i != root_j) {
            // Union by size
            if (size[root_i] < size[root_j])
                swap(root_i, root_j);
            
            parent[root_j] = root_i;
            size[root_i] += size[root_j];
            num_sets--;
            return true;
        }
        return false;
    }
    
    int get_size(int i) {
        return size[find(i)];
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    DSU dsu(n);
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        if (dsu.unite(u, v)) {
            cout << "Union " << u << " " << v << ": Merged\n";
        } else {
            cout << "Union " << u << " " << v << ": Already in same set\n";
        }
    }
    
    cout << "Number of connected components: " << dsu.num_sets << "\n";
    
    return 0;
}
