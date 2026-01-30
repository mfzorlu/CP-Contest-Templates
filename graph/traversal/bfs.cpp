#include <bits/stdc++.h>
using namespace std;

/*
 * BFS Template for Competitive Programming
 * Time Complexity: O(V + E)
 * Space Complexity: O(V)
 */

const int MAXN = 200005;
const int INF = 1e9;

vector<int> adj[MAXN];
int dist[MAXN];
int p[MAXN]; // Parent array for path reconstruction

void bfs(int start_node, int n) {
    // Initialize
    for (int i = 1; i <= n; i++) {
        dist[i] = -1;
        p[i] = -1;
    }
    
    queue<int> q;
    q.push(start_node);
    dist[start_node] = 0;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                p[v] = u;
                q.push(v);
            }
        }
    }
}

vector<int> get_path(int end_node) {
    if (dist[end_node] == -1) return {};
    
    vector<int> path;
    for (int v = end_node; v != -1; v = p[v]) {
        path.push_back(v);
    }
    reverse(path.begin(), path.end());
    return path;
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
    
    bfs(1, n);
    
    // Print distances
    for (int i = 1; i <= n; i++) {
        cout << dist[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
