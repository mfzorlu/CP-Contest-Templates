#include <iostream>
#include <vector>

using namespace std;

int knapsack(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            dp[i][w] = dp[i-1][w];
            
            if (weights[i-1] <= w) {
                int val = dp[i-1][w - weights[i-1]] + values[i-1];
                if (val > dp[i][w]) {
                    dp[i][w] = val;
                }
            }
        }
    }
    
    return dp[n][capacity];
}

int main() {
    return 0;
}
