#include <iostream>
#include <vector>

using namespace std;

const int INF = 1e9;

int solve_min_coins(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INF);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (i - coin >= 0) {
                if (dp[i - coin] != INF) {
                    if (dp[i - coin] + 1 < dp[i]) {
                        dp[i] = dp[i - coin] + 1;
                    }
                }
            }
        }
    }
    
    return dp[amount] == INF ? -1 : dp[amount];
}

int main() {
    return 0;
}
