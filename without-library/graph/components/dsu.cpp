#include <iostream>
#include <vector>
#include <numeric> // For iota (allowed, not "convenience" like algo adapters)
// Actually iota is in numeric, let's just write loop to be safe safe

using namespace std;

struct DSU {
    vector<int> parent;
    vector<int> size;
    
    DSU(int n) {
        parent.resize(n + 1);
        size.assign(n + 1, 1);
        for(int i=0; i<=n; i++) parent[i] = i;
    }
    
    int find(int i) {
        if(parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if(root_i != root_j) {
            if(size[root_i] < size[root_j]) {
                int temp = root_i; root_i = root_j; root_j = temp;
            }
            parent[root_j] = root_i;
            size[root_i] += size[root_j];
            return true;
        }
        return false;
    }
};

int main() {
    return 0;
}
