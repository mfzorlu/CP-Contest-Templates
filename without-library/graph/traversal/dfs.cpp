#include <iostream>
#include <vector>

using namespace std;

// DFS without bits/stdc++.h

const int MAXN = 200005;
vector<int> adj[MAXN];
bool visited[MAXN];
int entry_t[MAXN], exit_t[MAXN];
int timer;

void dfs(int u) {
    visited[u] = true;
    entry_t[u] = timer++;
    
    for (size_t i = 0; i < adj[u].size(); i++) {
        int v = adj[u][i];
        if (!visited[v]) {
            dfs(v);
        }
    }
    
    exit_t[u] = timer++;
}

int main() {
    return 0;
}
