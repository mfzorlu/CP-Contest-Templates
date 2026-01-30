#include <iostream>
#include <vector>

using namespace std;
typedef long long ll;

// Manual GCD
ll gcd_manual(ll a, ll b) {
    while (b) {
        ll temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Manual LCM
ll lcm_manual(ll a, ll b) {
    if (a == 0 || b == 0) return 0;
    return (a / gcd_manual(a, b)) * b;
}

// Extended GCD
ll extended_gcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    ll x1, y1;
    ll d = extended_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return d;
}

int main() {
    // Tests
    return 0;
}
