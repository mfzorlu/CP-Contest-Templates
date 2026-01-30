"""
Optimized GCD and LCM Algorithms for Competitive Programming
Time Complexity: O(log min(a, b)) for binary GCD, O(log(a+b)) for Euclidean
"""

import math
from functools import reduce

# ============= Basic GCD Algorithms =============

def gcd_euclidean(a, b):
    """
    Euclidean Algorithm - Classic and fast
    Time: O(log(min(a, b)))
    """
    while b:
        a, b = b, a % b
    return a


def gcd_recursive(a, b):
    """
    Recursive Euclidean Algorithm
    Time: O(log(min(a, b)))
    """
    if b == 0:
        return a
    return gcd_recursive(b, a % b)


def gcd_binary(a, b):
    """
    Binary GCD (Stein's Algorithm) - Faster for large numbers
    Uses bit operations instead of modulo
    Time: O(log min(a, b))
    """
    if a == 0:
        return b
    if b == 0:
        return a
    
    # Find common factors of 2
    shift = 0
    while ((a | b) & 1) == 0:
        shift += 1
        a >>= 1
        b >>= 1
    
    # Remove remaining factors of 2 from a
    while (a & 1) == 0:
        a >>= 1
    
    while b != 0:
        # Remove factors of 2 from b
        while (b & 1) == 0:
            b >>= 1
        
        # Swap if necessary
        if a > b:
            a, b = b, a
        
        b -= a
    
    return a << shift


# ============= Extended GCD =============

def extended_gcd(a, b):
    """
    Extended Euclidean Algorithm
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b)
    Used for modular inverse, Diophantine equations
    Time: O(log min(a, b))
    """
    if b == 0:
        return a, 1, 0
    
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd, x, y


def extended_gcd_iterative(a, b):
    """
    Iterative Extended GCD - Better for CP (no recursion limit)
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    return old_r, old_s, old_t  # gcd, x, y


# ============= LCM Algorithms =============

def lcm_basic(a, b):
    """
    LCM using GCD
    LCM(a, b) = (a * b) / GCD(a, b)
    Time: O(log min(a, b))
    """
    return abs(a * b) // gcd_euclidean(a, b)


def lcm_safe(a, b):
    """
    LCM with overflow protection
    Divides first to avoid overflow
    """
    return a // gcd_euclidean(a, b) * b


# ============= Multiple Numbers GCD/LCM =============

def gcd_multiple(numbers):
    """
    GCD of multiple numbers
    GCD(a, b, c) = GCD(GCD(a, b), c)
    """
    return reduce(gcd_euclidean, numbers)


def lcm_multiple(numbers):
    """
    LCM of multiple numbers
    LCM(a, b, c) = LCM(LCM(a, b), c)
    """
    return reduce(lcm_safe, numbers)


# ============= GCD Applications =============

def coprime_check(a, b):
    """
    Check if two numbers are coprime (GCD = 1)
    """
    return gcd_euclidean(a, b) == 1


def mod_inverse(a, m):
    """
    Modular multiplicative inverse of a modulo m
    Returns x such that (a * x) % m = 1
    Only exists if GCD(a, m) = 1
    """
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None  # Modular inverse doesn't exist
    return (x % m + m) % m


def solve_linear_diophantine(a, b, c):
    """
    Solve ax + by = c
    Returns (x0, y0) - one solution if exists, None otherwise
    General solution: x = x0 + k*(b/gcd), y = y0 - k*(a/gcd)
    """
    gcd, x, y = extended_gcd(a, b)
    
    if c % gcd != 0:
        return None  # No solution
    
    # Scale the solution
    x *= c // gcd
    y *= c // gcd
    
    return x, y


def gcd_range(a, b):
    """
    GCD of all numbers in range [a, b]
    GCD(a, a+1, a+2, ..., b) = 1 if b > a, else a
    """
    return a if a == b else 1


def lcm_factorial(n):
    """
    LCM of 1, 2, 3, ..., n
    Uses prime factorization approach
    """
    if n <= 0:
        return 1
    
    # Find all primes up to n
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(n)
    result = 1
    
    for p in primes:
        # Find highest power of p that divides n!
        power = 0
        pk = p
        while pk <= n:
            power += n // pk
            pk *= p
        result *= p ** power
    
    return result


# ============= Batch GCD (for multiple pairs) =============

def batch_gcd(pairs):
    """
    Compute GCD for multiple pairs efficiently
    """
    return [gcd_euclidean(a, b) for a, b in pairs]


# ============= GCD/LCM with Modulo =============

def gcd_mod(a, b, mod):
    """
    GCD in modular arithmetic
    """
    a %= mod
    b %= mod
    return gcd_euclidean(a, b)


# ============= CP Template - Quick Copy =============

def gcd(a, b):
    """Quick GCD for contests - use built-in"""
    return math.gcd(a, b)  # Python 3.5+


def lcm(a, b):
    """Quick LCM for contests"""
    return a * b // math.gcd(a, b)


# For Python 3.9+, you can use:
# from math import gcd, lcm


# ============= Testing and Examples =============

if __name__ == "__main__":
    # Basic tests
    a, b = 48, 18
    
    print(f"GCD({a}, {b}):")
    print(f"  Euclidean: {gcd_euclidean(a, b)}")
    print(f"  Binary: {gcd_binary(a, b)}")
    print(f"  Built-in: {math.gcd(a, b)}")
    
    print(f"\nLCM({a}, {b}): {lcm_basic(a, b)}")
    
    # Extended GCD
    gcd_val, x, y = extended_gcd(a, b)
    print(f"\nExtended GCD({a}, {b}):")
    print(f"  GCD = {gcd_val}")
    print(f"  {a}*{x} + {b}*{y} = {gcd_val}")
    print(f"  Verification: {a*x + b*y}")
    
    # Multiple numbers
    numbers = [12, 18, 24, 30]
    print(f"\nGCD of {numbers}: {gcd_multiple(numbers)}")
    print(f"LCM of {numbers}: {lcm_multiple(numbers)}")
    
    # Modular inverse
    a, m = 3, 11
    inv = mod_inverse(a, m)
    print(f"\nModular inverse of {a} mod {m}: {inv}")
    if inv:
        print(f"  Verification: ({a} * {inv}) mod {m} = {(a * inv) % m}")
    
    # Diophantine equation
    a, b, c = 12, 15, 9
    result = solve_linear_diophantine(a, b, c)
    if result:
        x, y = result
        print(f"\nSolution to {a}x + {b}y = {c}:")
        print(f"  x = {x}, y = {y}")
        print(f"  Verification: {a}*{x} + {b}*{y} = {a*x + b*y}")
    
    # Coprime check
    print(f"\nAre {a} and {b} coprime? {coprime_check(a, b)}")
    print(f"Are 15 and 28 coprime? {coprime_check(15, 28)}")
    
    # Performance comparison
    import time
    
    test_a, test_b = 123456789012345, 987654321098765
    
    start = time.time()
    for _ in range(100000):
        gcd_euclidean(test_a, test_b)
    time_euclidean = time.time() - start
    
    start = time.time()
    for _ in range(100000):
        gcd_binary(test_a, test_b)
    time_binary = time.time() - start
    
    print(f"\nPerformance (100k iterations with large numbers):")
    print(f"  Euclidean: {time_euclidean:.4f}s")
    print(f"  Binary: {time_binary:.4f}s")


"""
// C++ Version

/*
 * Optimized GCD and LCM Algorithms for Competitive Programming
 * Time Complexity: O(log min(a, b)) for binary GCD, O(log(a+b)) for Euclidean
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>

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


"""