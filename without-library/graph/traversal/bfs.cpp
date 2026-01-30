#include <iostream>
#include <vector>

using namespace std;

// BFS without std::queue

vector<int> adj[200005];
int dist[200005];
int parent[200005];

void bfs(int start, int n) {
    for(int i=0; i<=n; i++) dist[i] = -1;
    
    // Manual queue using vector
    vector<int> q;
    q.push_back(start);
    int head = 0;
    
    dist[start] = 0;
    parent[start] = -1;
    
    while(head < q.size()) {
        int u = q[head++];
        
        for(size_t i=0; i<adj[u].size(); i++) {
            int v = adj[u][i];
            if(dist[v] == -1) {
                dist[v] = dist[u] + 1;
                parent[v] = u;
                q.push_back(v);
            }
        }
    }
}

int main() {
    return 0;
}
