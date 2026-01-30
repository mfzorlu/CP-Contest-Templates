#include <bits/stdc++.h>
using namespace std;

/*
 * DP on Trees Template
 * Includes: Tree Diameter, Maximum Independent Set
 * Time Complexity: O(N)
 */

const int MAXN = 200005;
vector<int> adj[MAXN];
long long dp[MAXN][2]; // 0: without u, 1: with u

// Variables for Diameter
int max_dist = -1;
int farthest_node = -1;

void dfs_diameter(int u, int p, int d) {
    if (d > max_dist) {
        max_dist = d;
        farthest_node = u;
    }
    
    for (int v : adj[u]) {
        if (v != p) {
            dfs_diameter(v, u, d + 1);
        }
    }
}

int get_diameter(int n) {
    max_dist = -1;
    dfs_diameter(1, -1, 0);
    
    int start = farthest_node;
    max_dist = -1;
    dfs_diameter(start, -1, 0);
    
    return max_dist;
}

void dfs_dp(int u, int p) {
    dp[u][0] = 0;
    dp[u][1] = 1; // Weight of u
    
    for (int v : adj[u]) {
        if (v != p) {
            dfs_dp(v, u);
            
            // If we don't take u, we can either take v or not (greedy)
            dp[u][0] += max(dp[v][0], dp[v][1]);
            
            // If we take u, we cannot take v
            dp[u][1] += dp[v][0];
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    if (!(cin >> n)) return 0;
    
    // Tree has n-1 edges
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    cout << "Tree Diameter: " << get_diameter(n) << "\n";
    
    dfs_dp(1, -1);
    cout << "Max Independent Set Size: " << max(dp[1][0], dp[1][1]) << "\n";
    
    return 0;
}
