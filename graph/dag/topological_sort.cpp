#include <bits/stdc++.h>
using namespace std;

/*
 * Topological Sort (Kahn's Algorithm)
 * Time Complexity: O(V + E)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
int in_degree[MAXN];

vector<int> kahn(int n) {
    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }
    
    vector<int> topo_order;
    while (!q.empty()) {
        // Use priority_queue<int, vector<int>, greater<int>> for lexicographically smallest
        int u = q.front();
        q.pop();
        topo_order.push_back(u);
        
        for (int v : adj[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }
    }
    
    if (topo_order.size() < n) {
        return {}; // Cycle detected
    }
    
    return topo_order;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        in_degree[v]++;
    }
    
    vector<int> result = kahn(n);
    
    if (result.empty()) {
        cout << "IMPOSSIBLE\n";
    } else {
        for (int i = 0; i < n; i++) {
            cout << result[i] << (i == n - 1 ? "" : " ");
        }
        cout << "\n";
    }
    
    return 0;
}
