/*
 * Optimized GCD and LCM Algorithms for Competitive Programming
 * Time Complexity: O(log min(a, b)) for binary GCD, O(log(a+b)) for Euclidean
 */

#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

// ============= Basic GCD Algorithms =============

// Euclidean Algorithm - Most common in CP
ll gcd_euclidean(ll a, ll b) {
    while (b) {
        ll temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Recursive Euclidean (shorter code)
ll gcd_recursive(ll a, ll b) {
    return b == 0 ? a : gcd_recursive(b, a % b);
}

// One-liner using ternary
ll gcd_oneliner(ll a, ll b) {
    return b ? gcd_oneliner(b, a % b) : a;
}

// Binary GCD (Stein's Algorithm) - Faster for large numbers
ll gcd_binary(ll a, ll b) {
    if (a == 0) return b;
    if (b == 0) return a;
    
    // Find common power of 2
    int shift = 0;
    while (((a | b) & 1) == 0) {
        shift++;
        a >>= 1;
        b >>= 1;
    }
    
    // Remove remaining factors of 2 from a
    while ((a & 1) == 0) a >>= 1;
    
    while (b != 0) {
        while ((b & 1) == 0) b >>= 1;
        
        if (a > b) swap(a, b);
        b -= a;
    }
    
    return a << shift;
}

// ============= Extended GCD =============

// Extended Euclidean Algorithm - Returns gcd and coefficients
// Returns gcd(a, b) and finds x, y such that ax + by = gcd(a, b)
ll extended_gcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    
    ll x1, y1;
    ll gcd = extended_gcd(b, a % b, x1, y1);
    
    x = y1;
    y = x1 - (a / b) * y1;
    
    return gcd;
}

// Iterative Extended GCD - Better for CP (no stack overflow)
ll extended_gcd_iterative(ll a, ll b, ll &x, ll &y) {
    ll old_r = a, r = b;
    ll old_s = 1, s = 0;
    ll old_t = 0, t = 1;
    
    while (r != 0) {
        ll quotient = old_r / r;
        
        ll temp = r;
        r = old_r - quotient * r;
        old_r = temp;
        
        temp = s;
        s = old_s - quotient * s;
        old_s = temp;
        
        temp = t;
        t = old_t - quotient * t;
        old_t = temp;
    }
    
    x = old_s;
    y = old_t;
    return old_r;  // gcd
}

// ============= LCM Algorithms =============

// Basic LCM using GCD
ll lcm_basic(ll a, ll b) {
    return (a / gcd_euclidean(a, b)) * b;  // Divide first to avoid overflow
}

// LCM with __gcd (C++14+)
ll lcm_builtin(ll a, ll b) {
    return (a / __gcd(a, b)) * b;
}

// For C++17+, you can use std::lcm and std::gcd directly
// #include <numeric>
// ll result = lcm(a, b);

// ============= Multiple Numbers GCD/LCM =============

// GCD of array
ll gcd_array(vector<ll>& arr) {
    ll result = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        result = __gcd(result, arr[i]);
        if (result == 1) return 1;  // Early termination
    }
    return result;
}

// LCM of array
ll lcm_array(vector<ll>& arr) {
    ll result = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        result = (result / __gcd(result, arr[i])) * arr[i];
    }
    return result;
}

// GCD of range [a, b]
ll gcd_range(ll a, ll b) {
    return (a == b) ? a : 1;  // GCD of consecutive integers is always 1
}

// ============= GCD Applications =============

// Check if two numbers are coprime
bool coprime(ll a, ll b) {
    return __gcd(a, b) == 1;
}

// Modular multiplicative inverse using Extended GCD
// Returns x such that (a * x) % m = 1
ll mod_inverse(ll a, ll m) {
    ll x, y;
    ll gcd = extended_gcd_iterative(a, m, x, y);
    
    if (gcd != 1) return -1;  // Inverse doesn't exist
    
    return (x % m + m) % m;  // Make positive
}

// Solve linear Diophantine equation: ax + by = c
// Returns true if solution exists
bool solve_diophantine(ll a, ll b, ll c, ll &x, ll &y) {
    ll gcd = extended_gcd_iterative(a, b, x, y);
    
    if (c % gcd != 0) return false;
    
    // Scale the solution
    x *= c / gcd;
    y *= c / gcd;
    
    return true;
}

// Chinese Remainder Theorem (using GCD)
// Solve: x ≡ a1 (mod m1), x ≡ a2 (mod m2)
ll chinese_remainder(ll a1, ll m1, ll a2, ll m2) {
    ll x, y;
    ll gcd = extended_gcd_iterative(m1, m2, x, y);
    
    if ((a2 - a1) % gcd != 0) return -1;  // No solution
    
    ll lcm = m1 / gcd * m2;
    ll result = (a1 + m1 * x * ((a2 - a1) / gcd)) % lcm;
    
    return (result + lcm) % lcm;
}

// ============= GCD Properties & Tricks =============

// GCD of multiple pairs efficiently
vector<ll> batch_gcd(vector<pair<ll, ll>>& pairs) {
    vector<ll> results;
    for (auto& p : pairs) {
        results.push_back(__gcd(p.first, p.second));
    }
    return results;
}

// GCD with three numbers
ll gcd3(ll a, ll b, ll c) {
    return __gcd(__gcd(a, b), c);
}

// GCD sum: sum of GCD(i, n) for i = 1 to n
ll gcd_sum(ll n) {
    ll sum = 0;
    for (ll i = 1; i <= n; i++) {
        sum += __gcd(i, n);
    }
    return sum;
}

// Euler's Totient Function (counts numbers coprime to n)
ll euler_phi(ll n) {
    ll result = n;
    for (ll p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            while (n % p == 0) n /= p;
            result -= result / p;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}

// ============= CP Template - Quick Copy =============

// Most compact GCD for contests (C++14+)
#define gcd __gcd

// For LCM
ll lcm(ll a, ll b) {
    return a / __gcd(a, b) * b;
}

// One-liner recursive GCD (if __gcd not available)
ll g(ll a, ll b) { return b ? g(b, a % b) : a; }

// ============= Precomputed GCD Table =============

const int MAXN = 1001;
int gcd_table[MAXN][MAXN];

void precompute_gcd() {
    for (int i = 1; i < MAXN; i++) {
        for (int j = 1; j < MAXN; j++) {
            gcd_table[i][j] = __gcd(i, j);
        }
    }
}

// ============= Main - Testing and Benchmarking =============

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Basic tests
    ll a = 48, b = 18;
    
    cout << "GCD(" << a << ", " << b << "):\n";
    cout << "  Euclidean: " << gcd_euclidean(a, b) << "\n";
    cout << "  Binary: " << gcd_binary(a, b) << "\n";
    cout << "  Built-in __gcd: " << __gcd(a, b) << "\n\n";
    
    cout << "LCM(" << a << ", " << b << "): " << lcm_basic(a, b) << "\n\n";
    
    // Extended GCD
    ll x, y;
    ll gcd_val = extended_gcd_iterative(a, b, x, y);
    cout << "Extended GCD(" << a << ", " << b << "):\n";
    cout << "  GCD = " << gcd_val << "\n";
    cout << "  " << a << "*" << x << " + " << b << "*" << y << " = " << gcd_val << "\n";
    cout << "  Verification: " << (a*x + b*y) << "\n\n";
    
    // Multiple numbers
    vector<ll> numbers = {12, 18, 24, 30};
    cout << "GCD of array: " << gcd_array(numbers) << "\n";
    cout << "LCM of array: " << lcm_array(numbers) << "\n\n";
    
    // Modular inverse
    ll a_mod = 3, m = 11;
    ll inv = mod_inverse(a_mod, m);
    cout << "Modular inverse of " << a_mod << " mod " << m << ": " << inv << "\n";
    if (inv != -1) {
        cout << "  Verification: (" << a_mod << " * " << inv << ") mod " << m 
             << " = " << (a_mod * inv) % m << "\n\n";
    }
    
    // Diophantine equation
    ll a_dio = 12, b_dio = 15, c_dio = 9;
    ll x_dio, y_dio;
    if (solve_diophantine(a_dio, b_dio, c_dio, x_dio, y_dio)) {
        cout << "Solution to " << a_dio << "x + " << b_dio << "y = " << c_dio << ":\n";
        cout << "  x = " << x_dio << ", y = " << y_dio << "\n";
        cout << "  Verification: " << (a_dio*x_dio + b_dio*y_dio) << "\n\n";
    }
    
    // Coprime check
    cout << "Are " << a << " and " << b << " coprime? " << (coprime(a, b) ? "Yes" : "No") << "\n";
    cout << "Are 15 and 28 coprime? " << (coprime(15, 28) ? "Yes" : "No") << "\n\n";
    
    // Euler's totient
    ll n = 12;
    cout << "Euler's totient φ(" << n << ") = " << euler_phi(n) << "\n\n";
    
    // Performance comparison
    ll test_a = 123456789012345LL;
    ll test_b = 987654321098765LL;
    int iterations = 1000000;
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        gcd_euclidean(test_a, test_b);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration_euclidean = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        gcd_binary(test_a, test_b);
    }
    end = chrono::high_resolution_clock::now();
    auto duration_binary = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        __gcd(test_a, test_b);
    }
    end = chrono::high_resolution_clock::now();
    auto duration_builtin = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "Performance (" << iterations << " iterations with large numbers):\n";
    cout << "  Euclidean: " << duration_euclidean.count() << "ms\n";
    cout << "  Binary: " << duration_binary.count() << "ms\n";
    cout << "  Built-in __gcd: " << duration_builtin.count() << "ms\n";
    
    return 0;
}
