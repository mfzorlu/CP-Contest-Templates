#include <iostream>
#include <vector>

using namespace std;

vector<int> adj[200005];
int memo[200005];

int longest_path(int u) {
    if(memo[u] != -1) return memo[u];
    
    int mx = 0;
    for(size_t i=0; i<adj[u].size(); i++) {
        int v = adj[u][i];
        int val = 1 + longest_path(v);
        if(val > mx) mx = val;
    }
    
    return memo[u] = mx;
}

int main() {
    return 0;
}
