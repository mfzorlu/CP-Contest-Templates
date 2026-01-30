#include <bits/stdc++.h>
using namespace std;

/*
 * DFS Template for Competitive Programming
 * Time Complexity: O(V + E)
 * Space Complexity: O(V)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
bool visited[MAXN];
int entry[MAXN], leave[MAXN];
int timer;

void dfs(int u) {
    visited[u] = true;
    entry[u] = timer++;
    
    for (int v : adj[u]) {
        if (!visited[v]) {
            dfs(v);
        }
    }
    
    leave[u] = timer++;
}

void dfs_iterative(int start_node) {
    stack<int> s;
    s.push(start_node);
    
    while (!s.empty()) {
        int u = s.top();
        s.pop();
        
        if (!visited[u]) {
            visited[u] = true;
            // Process node u
            
            for (auto it = adj[u].rbegin(); it != adj[u].rend(); ++it) {
                if (!visited[*it]) {
                    s.push(*it);
                }
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
        adj[v].push_back(u); // Undirected
    }
    
    // Reset
    fill(visited, visited + n + 1, false);
    timer = 0;
    
    int components = 0;
    for (int i = 1; i <= n; i++) {
        if (!visited[i]) {
            components++;
            dfs(i);
        }
    }
    
    cout << "Connected Components: " << components << "\n";
    
    return 0;
}
