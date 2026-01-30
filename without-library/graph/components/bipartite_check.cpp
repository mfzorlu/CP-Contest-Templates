#include <iostream>
#include <vector>

using namespace std;

vector<int> adj[200005];
int color[200005];

bool check_bipartite(int start) {
    vector<int> q;
    q.push_back(start);
    int head = 0;
    color[start] = 0;
    
    while(head < q.size()) {
        int u = q[head++];
        
        for(size_t i=0; i<adj[u].size(); i++) {
            int v = adj[u][i];
            if(color[v] == -1) {
                color[v] = 1 - color[u];
                q.push_back(v);
            } else if(color[v] == color[u]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    return 0;
}
