"""
Optimized Sieve of Eratosthenes for Competitive Programming
Time Complexity: O(n log log n)
Space Complexity: O(n)
"""

def sieve_of_eratosthenes(n):
    """
    Standard optimized sieve - finds all primes up to n
    """
    if n < 2:
        return []
    
    # Only store odd numbers (except 2)
    # is_prime[i] represents whether 2*i+3 is prime
    is_prime = [True] * ((n - 1) // 2)
    limit = int(n ** 0.5)
    
    for i in range(limit // 2):
        if is_prime[i]:
            # The actual prime number
            p = 2 * i + 3
            # Mark multiples as composite, starting from p*p
            # Only mark odd multiples
            start = (p * p - 3) // 2
            for j in range(start, len(is_prime), p):
                is_prime[j] = False
    
    # Collect all primes
    primes = [2] + [2 * i + 3 for i in range(len(is_prime)) if is_prime[i]]
    return primes


def simple_sieve(limit):
    """
    Simple sieve for small limits (used in segmented sieve)
    """
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]


def segmented_sieve(n):
    """
    Segmented Sieve - Memory efficient for very large n
    Time Complexity: O(n log log n)
    Space Complexity: O(sqrt(n))
    """
    if n < 2:
        return []
    
    limit = int(n ** 0.5) + 1
    base_primes = simple_sieve(limit)
    
    primes = base_primes.copy()
    
    # Process in segments
    segment_size = max(limit, 32768)  # Cache-friendly size
    low = limit + 1
    
    while low <= n:
        high = min(low + segment_size - 1, n)
        
        # Create a segment
        is_prime = [True] * (high - low + 1)
        
        # Use base primes to mark composites in this segment
        for p in base_primes:
            # Find the minimum number in [low, high] that is a multiple of p
            start = max(p * p, ((low + p - 1) // p) * p)
            
            for j in range(start, high + 1, p):
                is_prime[j - low] = False
        
        # Collect primes from this segment
        primes.extend([low + i for i in range(len(is_prime)) if is_prime[i]])
        
        low += segment_size
    
    return primes


def linear_sieve(n):
    """
    Linear Sieve (Sieve of Euler) - O(n) time complexity!
    Each composite number is visited exactly once
    Also computes smallest prime factor (spf)
    """
    if n < 2:
        return []
    
    spf = [0] * (n + 1)  # Smallest prime factor
    primes = []
    
    for i in range(2, n + 1):
        if spf[i] == 0:  # i is prime
            spf[i] = i
            primes.append(i)
        
        # Mark multiples using existing primes
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            spf[i * p] = p
    
    return primes


# Template for CP contests
def get_primes_fast(n):
    """
    Quick template for contests - choose based on constraints
    For n <= 10^7: use standard sieve
    For n <= 10^9: use segmented sieve
    For n <= 10^6 and need SPF: use linear sieve
    """
    if n <= 10_000_000:
        return sieve_of_eratosthenes(n)
    else:
        return segmented_sieve(n)


# Example usage and testing
if __name__ == "__main__":
    # Test with different methods
    n = 100
    
    print(f"Primes up to {n}:")
    print("Standard Sieve:", sieve_of_eratosthenes(n))
    print("\nSegmented Sieve:", segmented_sieve(n))
    print("\nLinear Sieve:", linear_sieve(n))
    
    # Performance test
    import time
    
    test_n = 1_000_000
    
    start = time.time()
    primes1 = sieve_of_eratosthenes(test_n)
    print(f"\nStandard Sieve for n={test_n}: {time.time() - start:.4f}s, found {len(primes1)} primes")
    
    start = time.time()
    primes2 = linear_sieve(test_n)
    print(f"Linear Sieve for n={test_n}: {time.time() - start:.4f}s, found {len(primes2)} primes")




"""
// C++ Version

/*
 * Optimized Sieve of Eratosthenes for Competitive Programming
 * Time Complexity: O(n log log n)
 * Space Complexity: O(n)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include <chrono>

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

// ============= Main - Testing and Benchmarking =============
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n = 100;
    
    cout << "Primes up to " << n << ":\n";
    vector<int> primes = sieve_of_eratosthenes(n);
    for (int p : primes) {
        cout << p << " ";
    }
    cout << "\n\n";
    
    // Performance test
    int test_n = 1000000;
    
    auto start = chrono::high_resolution_clock::now();
    vector<int> primes1 = sieve_of_eratosthenes(test_n);
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    start = chrono::high_resolution_clock::now();
    vector<int> primes2 = linear_sieve(test_n);
    end = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "Standard Sieve for n=" << test_n << ": " 
         << duration1.count() << "ms, found " << primes1.size() << " primes\n";
    cout << "Linear Sieve for n=" << test_n << ": " 
         << duration2.count() << "ms, found " << primes2.size() << " primes\n";
    
    return 0;
}



"""