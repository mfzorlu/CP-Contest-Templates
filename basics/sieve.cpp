/*
 * Optimized Sieve of Eratosthenes for Competitive Programming
 * Time Complexity: O(n log log n)
 * Space Complexity: O(n)
 */

#include <bits/stdc++.h>

using namespace std;

// ============= Standard Optimized Sieve =============
vector<int> sieve_of_eratosthenes(int n) {
    if (n < 2) return {};
    
    // Only store odd numbers (2 is handled separately)
    vector<bool> is_prime((n - 1) / 2, true);
    int limit = sqrt(n);
    
    for (int i = 0; i < limit / 2; i++) {
        if (is_prime[i]) {
            int p = 2 * i + 3;
            // Mark multiples starting from p*p
            for (int j = (p * p - 3) / 2; j < is_prime.size(); j += p) {
                is_prime[j] = false;
            }
        }
    }
    
    vector<int> primes = {2};
    for (int i = 0; i < is_prime.size(); i++) {
        if (is_prime[i]) {
            primes.push_back(2 * i + 3);
        }
    }
    
    return primes;
}

// ============= Bitset Sieve (Memory Efficient) =============
const int MAXN = 100000000; // Adjust based on problem
bitset<MAXN> is_composite;

vector<int> bitset_sieve(int n) {
    is_composite.reset();
    is_composite[0] = is_composite[1] = 1;
    
    for (int i = 2; i * i <= n; i++) {
        if (!is_composite[i]) {
            for (long long j = (long long)i * i; j <= n; j += i) {
                is_composite[j] = 1;
            }
        }
    }
    
    vector<int> primes;
    for (int i = 2; i <= n; i++) {
        if (!is_composite[i]) {
            primes.push_back(i);
        }
    }
    
    return primes;
}

// ============= Segmented Sieve (For Very Large N) =============
vector<int> simple_sieve(int limit) {
    vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    for (int i = 2; i * i <= limit; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }
    
    vector<int> primes;
    for (int i = 2; i <= limit; i++) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}

vector<int> segmented_sieve(long long n) {
    if (n < 2) return {};
    
    int limit = sqrt(n) + 1;
    vector<int> base_primes = simple_sieve(limit);
    vector<int> primes = base_primes;
    
    int segment_size = max(limit, 32768); // Cache-friendly
    vector<bool> is_prime(segment_size);
    
    for (long long low = limit + 1; low <= n; low += segment_size) {
        fill(is_prime.begin(), is_prime.end(), true);
        
        long long high = min(low + segment_size - 1, n);
        
        for (int p : base_primes) {
            long long start = max((long long)p * p, ((low + p - 1) / p) * p);
            
            for (long long j = start; j <= high; j += p) {
                is_prime[j - low] = false;
            }
        }
        
        for (long long i = low; i <= high; i++) {
            if (is_prime[i - low]) {
                primes.push_back(i);
            }
        }
    }
    
    return primes;
}

// ============= Linear Sieve (O(n) - Fastest!) =============
vector<int> linear_sieve(int n) {
    if (n < 2) return {};
    
    vector<int> spf(n + 1, 0); // Smallest prime factor
    vector<int> primes;
    
    for (int i = 2; i <= n; i++) {
        if (spf[i] == 0) { // i is prime
            spf[i] = i;
            primes.push_back(i);
        }
        
        for (int p : primes) {
            if (p > spf[i] || (long long)i * p > n) break;
            spf[i * p] = p;
        }
    }
    
    return primes;
}

// ============= CP Template - Quick Copy =============
// Choose based on constraints:
// n <= 10^7: standard sieve
// n <= 10^8: bitset sieve
// n <= 10^12: segmented sieve
// Need SPF or factorization: linear sieve

vector<int> get_primes(int n) {
    if (n <= 10000000) {
        return sieve_of_eratosthenes(n);
    } else if (n <= 100000000) {
        return bitset_sieve(n);
    } else {
        return segmented_sieve(n);
    }
}

// ============= Utility Functions =============
// Check if n is prime using sieve
bool is_prime_check(int n, const vector<int>& primes) {
    if (n < 2) return false;
    for (int p : primes) {
        if (p * p > n) break;
        if (n % p == 0) return false;
    }
    return true;
}

// Count primes up to n (using linear sieve with counting)
int count_primes(int n) {
    if (n < 2) return 0;
    
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    int count = 0;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            count++;
            if ((long long)i * i <= n) {
                for (int j = i * i; j <= n; j += i) {
                    is_prime[j] = false;
                }
            }
        }
    }
    return count;
}
