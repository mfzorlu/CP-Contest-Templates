#include <bits/stdc++.h>
using namespace std;

/*
 * Cycle Detection Template
 * Supports both Directed and Undirected graphs
 * Time Complexity: O(V + E)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
bool visited[MAXN];
bool recursion_stack[MAXN]; // for directed
int parent[MAXN];
int cycle_start = -1, cycle_end = -1;

// Undirected Cycle Detection
bool dfs_undirected(int u, int p) {
    visited[u] = true;
    parent[u] = p;
    
    for (int v : adj[u]) {
        if (v == p) continue;
        if (visited[v]) {
            cycle_end = u;
            cycle_start = v;
            return true;
        }
        if (dfs_undirected(v, u)) return true;
    }
    return false;
}

// Directed Cycle Detection
bool dfs_directed(int u) {
    visited[u] = true;
    recursion_stack[u] = true;
    
    for (int v : adj[u]) {
        if (!visited[v]) {
            parent[v] = u;
            if (dfs_directed(v)) return true;
        } else if (recursion_stack[v]) {
            cycle_end = u;
            cycle_start = v;
            return true;
        }
    }
    
    recursion_stack[u] = false;
    return false;
}

void print_cycle() {
    if (cycle_start == -1) {
        cout << "No cycle found\n";
        return;
    }
    
    vector<int> cycle;
    cycle.push_back(cycle_start);
    for (int v = cycle_end; v != cycle_start; v = parent[v]) {
        cycle.push_back(v);
    }
    cycle.push_back(cycle_start);
    reverse(cycle.begin(), cycle.end());
    
    cout << "Cycle found: ";
    for (int v : cycle) cout << v << " ";
    cout << "\n";
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    bool is_directed = false; // Toggle manually
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        if (!is_directed) adj[v].push_back(u);
    }
    
    fill(visited, visited + n + 1, false);
    fill(parent, parent + n + 1, -1);
    
    bool cycle_found = false;
    for (int i = 1; i <= n; i++) {
        if (!visited[i]) {
            if (is_directed) {
                if (dfs_directed(i)) {
                    cycle_found = true;
                    break;
                }
            } else {
                if (dfs_undirected(i, -1)) {
                    cycle_found = true;
                    break;
                }
            }
        }
    }
    
    if (cycle_found) print_cycle();
    else cout << "No cycle\n";
    
    return 0;
}
