#include <iostream>
#include <vector>

using namespace std;

vector<int> adj[200005];
bool visited[200005];
bool rec_stack[200005];

bool dfs_undirected(int u, int p) {
    visited[u] = true;
    for(size_t i=0; i<adj[u].size(); i++) {
        int v = adj[u][i];
        if(v == p) continue;
        if(visited[v]) return true;
        if(dfs_undirected(v, u)) return true;
    }
    return false;
}

bool dfs_directed(int u) {
    visited[u] = true;
    rec_stack[u] = true;
    for(size_t i=0; i<adj[u].size(); i++) {
        int v = adj[u][i];
        if(!visited[v]) {
            if(dfs_directed(v)) return true;
        } else if(rec_stack[v]) {
            return true;
        }
    }
    rec_stack[u] = false;
    return false;
}

int main() {
    return 0;
}
