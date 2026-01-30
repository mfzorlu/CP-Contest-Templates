#include <iostream>
#include <vector>
#include <algorithm> // for max

using namespace std;

vector<int> adj[200005];
long long dp[200005][2];

void dfs_dp(int u, int p) {
    dp[u][0] = 0;
    dp[u][1] = 1;
    
    for(size_t i=0; i<adj[u].size(); i++) {
        int v = adj[u][i];
        if(v != p) {
            dfs_dp(v, u);
            dp[u][0] += max(dp[v][0], dp[v][1]);
            dp[u][1] += dp[v][0];
        }
    }
}

int main() {
    return 0;
}
