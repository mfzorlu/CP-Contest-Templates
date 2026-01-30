#include <bits/stdc++.h>
using namespace std;

/*
 * Bipartite Check Template
 * Checks if graph can be colored with 2 colors
 * Time Complexity: O(V + E)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
int color[MAXN]; // -1: uncolored, 0, 1
bool is_bipartite = true;

void bfs_check(int start_node) {
    queue<int> q;
    q.push(start_node);
    color[start_node] = 0;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v : adj[u]) {
            if (color[v] == -1) {
                color[v] = 1 - color[u];
                q.push(v);
            } else if (color[v] == color[u]) {
                is_bipartite = false;
                return;
            }
        }
    }
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
        adj[v].push_back(u);
    }
    
    fill(color, color + n + 1, -1);
    
    for (int i = 1; i <= n; i++) {
        if (color[i] == -1) {
            bfs_check(i);
            if (!is_bipartite) break;
        }
    }
    
    if (is_bipartite) cout << "YES\n";
    else cout << "NO\n";
    
    return 0;
}
