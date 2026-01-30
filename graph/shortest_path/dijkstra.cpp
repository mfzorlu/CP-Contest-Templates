#include <bits/stdc++.h>
using namespace std;

/*
 * Dijkstra's Algorithm
 * Shortest Path with non-negative weights
 * Time Complexity: O(E log V)
 */

const int MAXN = 200005;
const long long INF = 1e18; // Use long long for distances

struct Edge {
    int to;
    int weight;
};

vector<Edge> adj[MAXN];
long long dist[MAXN];
int parent[MAXN];

void dijkstra(int start_node, int n) {
    for (int i = 1; i <= n; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    
    dist[start_node] = 0;
    
    // min-priority queue: (distance, node)
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
    pq.push({0, start_node});
    
    while (!pq.empty()) {
        long long d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        // Important: Lazy deletion check
        if (d > dist[u]) continue;
        
        for (auto& edge : adj[u]) {
            int v = edge.to;
            int w = edge.weight;
            
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
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
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected
    }
    
    dijkstra(1, n);
    
    for (int i = 1; i <= n; i++) {
        if (dist[i] == INF) cout << "INF ";
        else cout << dist[i] << " ";
    }
    cout << "\n";
    
    return 0;
}
