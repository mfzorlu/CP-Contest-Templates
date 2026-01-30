#include <iostream>
#include <vector>

using namespace std;

vector<int> adj[200005];
int in_degree[200005];

vector<int> kahn(int n) {
    for(int i=1; i<=n; i++) {
        for(size_t j=0; j<adj[i].size(); j++) {
            in_degree[adj[i][j]]++;
        }
    }
    
    vector<int> q;
    for(int i=1; i<=n; i++) {
        if(in_degree[i] == 0) q.push_back(i);
    }
    
    int head = 0;
    vector<int> topo;
    
    while(head < q.size()) {
        int u = q[head++];
        topo.push_back(u);
        
        for(size_t i=0; i<adj[u].size(); i++) {
            int v = adj[u][i];
            in_degree[v]--;
            if(in_degree[v] == 0) q.push_back(v);
        }
    }
    
    if(topo.size() < n) return {};
    return topo;
}

int main() {
    return 0;
}
