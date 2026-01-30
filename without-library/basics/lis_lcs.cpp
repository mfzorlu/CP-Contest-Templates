#include <iostream>
#include <vector>
#include <string>

using namespace std;

int lis(vector<int>& arr) {
    if (arr.empty()) return 0;
    int n = arr.size();
    vector<int> dp(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[i] > arr[j]) {
                if (dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
    }
    
    int mx = 0;
    for (int x : dp) if (x > mx) mx = x;
    return mx;
}

int lcs(string s1, string s2) {
    int n = s1.length();
    int m = s2.length();
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                int v1 = dp[i-1][j];
                int v2 = dp[i][j-1];
                dp[i][j] = (v1 > v2) ? v1 : v2;
            }
        }
    }
    
    return dp[n][m];
}

int main() {
    return 0;
}
