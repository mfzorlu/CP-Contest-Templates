#include <bits/stdc++.h>
using namespace std;

/*
 * DP on DAG (Directed Acyclic Graph)
 * Example: Longest Path
 * Time Complexity: O(V + E)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
int memo[MAXN];
int parent[MAXN]; // For path reconstruction

int longest_path(int u) {
    if (memo[u] != -1) return memo[u];
    
    int max_len = 0;
    for (int v : adj[u]) {
        int val = 1 + longest_path(v);
        if (val > max_len) {
            max_len = val;
            parent[u] = v;
        }
    }
    
    return memo[u] = max_len;
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
    }
    
    fill(memo, memo + n + 1, -1);
    fill(parent, parent + n + 1, -1);
    
    int global_max = 0;
    int global_start = -1;
    
    for (int i = 1; i <= n; i++) {
        int val = longest_path(i);
        if (val > global_max) {
            global_max = val;
            global_start = i;
        }
    }
    
    cout << "Longest Path Length: " << global_max << "\n";
    
    return 0;
}
