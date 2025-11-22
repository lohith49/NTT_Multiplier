#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;
#include <omp.h> 
#include <immintrin.h>
using cd = complex<double>;
const double PI = acos(-1);

// Utility functions
int nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

bool isPowerOf2(int n) {
    return n && (!(n & (n - 1)));
}

bool isPowerOf4(int n) {
    if (n == 0) return false;
    while (n != 1) {
        if (n % 4 != 0) return false;
        n = n / 4;
    }
    return true;
}

int distanceToNextPower2(int n) {
    return nextPowerOf2(n) - n;
}

double calculateSparsity(const vector<cd>& polynomial) {
    int zeros = 0;
    for (const auto& coeff : polynomial) {
        if (abs(coeff) < 1e-10) zeros++;
    }
    return static_cast<double>(zeros) / polynomial.size();
}

// Algorithm 1: Radix-2 FFT
void fft_radix_2(vector<cd>& A, vector<cd>& X) {
    int n = A.size();
    if (n == 1) return;

    vector<cd>A_even(n/2);
    vector<cd>A_odd(n/2);
    for (int i = 0; i < n/2; i++) {
        A_even[i] = A[2*i];
        A_odd[i] = A[2*i+1];
    }

    vector<cd> X_2(n);

    for (int i = 0; i < n; i++)
        X_2[i] = X[i] * X[i];

    fft_radix_2(A_even, X_2);
    fft_radix_2(A_odd, X_2);


    for (int i = 0; i < n/2; i++) {
        A[i]         = A_even[i] + X[i] * A_odd[i];
        A[i + n/2]   = A_even[i] + X[i + n/2] * A_odd[i];
    }
}

// Algorithm 2: Modified Radix-4 FFT
void fft_m_radix_4(vector<cd> &A, vector<cd>& X) {
    int n = A.size();
    if (n == 1) return;
    if (n == 2) {
        cd a = A[0];
        cd b = A[1];
        A[0] = a + b;
        A[1] = a - b;
        return;
    }

    vector<cd> A0(n/4), A1(n/4), A2(n/4), A3(n/4);
    for (int i = 0; i < n/4; i++) {
        A0[i] = A[4*i];
        A1[i] = A[4*i+1];
        A2[i] = A[4*i+2];
        A3[i] = A[4*i+3];
    }

    vector<cd> X4(n);

    for (int i = 0; i < n; i++) {
        cd x2 = X[i] * X[i];
        X4[i] = x2 * x2;
    }

    fft_m_radix_4(A0, X4);
    fft_m_radix_4(A1, X4);
    fft_m_radix_4(A2, X4);
    fft_m_radix_4(A3, X4);


    for (int i = 0; i < n/4; i++) {
        A[i] = A0[i] + X[i]*A1[i] + X[(2*i)%n]*A2[i] + X[(3*i)%n]*A3[i];
    }

    for (int i = 0; i < n/4; i++) {
        int idx = i + n/4;
        A[idx] = A0[i] + X[idx]*A1[i] + X[(2*idx)%n]*A2[i] + X[(3*idx)%n]*A3[i];
    }

    for (int i = 0; i < n/4; i++) {
        int idx = i + n/2;
        A[idx] = A0[i] + X[idx]*A1[i] + X[(2*idx)%n]*A2[i] + X[(3*idx)%n]*A3[i];
    }


    for (int i = 0; i < n/4; i++) {
        int idx = i + 3*(n/4);
        A[idx] = A0[i] + X[idx]*A1[i] + X[(2*idx)%n]*A2[i] + X[(3*idx)%n]*A3[i];
    }
}

// Algorithm 3: Radix Split FFT
void fft_radix_split(vector<cd> &A, vector<cd>& X) {
    int n = A.size();
    if (n <= 1) return;

    if (n == 2) {
        cd a = A[0], b = A[1];
        A[0] = a + b;
        A[1] = a - b;
        return;
    }

    vector<cd> A0(n / 2), A1(n / 4), A3(n / 4);

    for (int i = 0; i < n / 2; i++)
        A0[i] = A[2 * i];
    

    for (int i = 0; i < n / 4; i++) {
        A1[i] = A[4 * i + 1];
        A3[i] = A[4 * i + 3];
    }

    // Precompute powers
    vector<cd> X2(n), X4(n);

    for (int i = 0; i < n; i++) {
        X2[i] = X[i] * X[i];
        X4[i] = X2[i] * X2[i];
    }

    fft_radix_split(A0, X2);
    fft_radix_split(A1, X4);
    fft_radix_split(A3, X4);


    // Merge step

    for (int i = 0; i < n / 4; i++) {
        A[i]=A0[i]+X[i]*A1[i]+X[(3*i)%n]*A3[i];
    }

    for (int i = 0; i < n / 4; i++) {
        A[i+(n/4)]=A0[i+(n/4)]+X[i+n/4]*A1[i]+X[(3*(i+n/4))%n]*A3[i];
    }

    for (int i = 0; i < n / 4; i++) {
        A[(i+(n/2))]=A0[i]+X[i+n/2]*A1[i]+X[(3*(i+n/2))%n]*A3[i];
    }

    for (int i = 0; i < n / 4; i++) {
        A[(i+3*(n/4))]=A0[i+(n/4)]+X[i+3*n/4]*A1[i]+X[(3*(i+3*n/4))%n]*A3[i];
    }
}
// Test function for measuring performance
struct TestResult {
    int polynomialSize;
    int paddedSize;
    double sparsity;
    int distToNextPow2;
    bool isPow2;
    bool isPow4;
    double time_radix_2;
    double time_m_radix_4;
    double time_radix_split;
    string bestAlgorithm;
};

TestResult runPerformanceTest(int polySize,double targetSparsity) {
    TestResult result;
    result.polynomialSize = polySize;
    result.distToNextPow2 = distanceToNextPower2(polySize);
    result.isPow2 = isPowerOf2(polySize);
    result.isPow4 = isPowerOf4(polySize);
    
    // Generate random polynomial
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100, 100);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
  
         int zr = targetSparsity*polySize;

    vector<cd> original_poly(polySize);
         for(int i=0;i<zr;i++)
         {
            original_poly[i]=0;
         }
         for(int i=zr;i<polySize;i++)
         {
            original_poly[i]=dist(gen);
            }
    result.sparsity = targetSparsity;

    shuffle(original_poly.begin(), original_poly.end(), gen);
    
    // Prepare the test data once
    int x = nextPowerOf2(original_poly.size());
    result.paddedSize = x;
    
    vector<cd> X(x);
    double ang = 2 * PI / x;
    for (int i = 0; i < x; i++) {
        X[i] = cd(cos(ang * i), sin(ang * i));
    }

    // Test Radix-2 FFT
    {
        vector<cd> a = original_poly;
        a.resize(x);
        
        auto start = high_resolution_clock::now();
        fft_radix_2(a, X);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        result.time_radix_2 = duration.count() / 1000.0; // Convert to milliseconds
    }
    
    // Test Modified Radix-4 FFT
    {
        vector<cd> a = original_poly;
        a.resize(x);
        
        auto start = high_resolution_clock::now();
        fft_m_radix_4(a, X);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        result.time_m_radix_4 = duration.count() / 1000.0;
    }
    
    // Test Radix Split FFT
    {
        vector<cd> a = original_poly;
        a.resize(x);
        
        auto start = high_resolution_clock::now();
        fft_radix_split(a, X);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        result.time_radix_split = duration.count() / 1000.0;
    }
    
    // Determine best algorithm
    double min_time = min({result.time_radix_2, result.time_m_radix_4, result.time_radix_split});
    if (min_time == result.time_radix_2) {
        result.bestAlgorithm = "Radix-2";
    } else if (min_time == result.time_m_radix_4) {
        result.bestAlgorithm = "Modified-Radix-4";
    } else {
        result.bestAlgorithm = "Radix-Split";
    }
    
    return result;
}

signed main() {
    cout << "Enhanced FFT Performance Comparison Test\n";
    cout << "======================================\n\n";
    
    // Open CSV file for output
    ofstream csvFile("fft_performance_results.csv");
    csvFile << "Test_ID,Polynomial_Size,Padded_Size,Sparsity,Dist_To_Next_Pow2,Is_Power_2,Is_Power_4,"
            << "Radix_2_Time_ms,Modified_Radix_4_Time_ms,Radix_Split_Time_ms,Best_Algorithm\n";
    
    vector<TestResult> results;
    vector<double> sparsityLevels;
    int numSparsities = 10;
    for (int i = 0; i < numSparsities; ++i) {
        sparsityLevels.push_back(i / double(numSparsities - 1));  // includes 0.0 and 1.0
    }
    int test_id= 1;
    // Generate 200 test cases with varying polynomial sizes
    for (int i = 1; i <= 5000; i++) {
        int polySize=i;
            for (double targetSparsity : sparsityLevels) {
        cout << "Running test " << test_id << "/" << numSparsities*5000
             << ", Target Sparsity: " << targetSparsity << " ... ";
        cout.flush();

        // Generate polynomial with approximate sparsity
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(-100, 100);
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        // Save this polynomial to file (optional)

        // Run FFT performance test on this polynomial
        TestResult result = runPerformanceTest(polySize,targetSparsity);

        results.push_back(result);

        csvFile << test_id << ","
                << result.polynomialSize << ","
                << result.paddedSize << ","
                << result.sparsity << ","
                << result.distToNextPow2 << ","
                << result.isPow2 << ","
                << result.isPow4 << ","
                << result.time_radix_2 << ","
                << result.time_m_radix_4 << ","
                << result.time_radix_split << ","
                << result.bestAlgorithm << "\n";

        cout << "Done\n";
        test_id++;
    }
    }
    
    csvFile.close();
    
    // Print summary statistics
    cout << "\nTest completed! Results saved to fft_performance_results.csv\n";
    cout << "Summary Statistics:\n";
    cout << "==================\n";
    
    double avg_radix_2 = 0, avg_m_radix_4 = 0, avg_radix_split = 0;
    double avg_sparsity = 0;
    map<string, int> best_algo_count;
    
    for (const auto& result : results) {
        avg_radix_2 += result.time_radix_2;
        avg_m_radix_4 += result.time_m_radix_4;
        avg_radix_split += result.time_radix_split;
        avg_sparsity += result.sparsity;
        best_algo_count[result.bestAlgorithm]++;
    }
    
    int n = results.size();
    avg_radix_2 /= n;
    avg_m_radix_4 /= n;
    avg_radix_split /= n;
    avg_sparsity /= n;
    
    cout << "Average execution times (ms):\n";
    cout << "Radix-2 FFT:         " << avg_radix_2 << "\n";
    cout << "Modified Radix-4 FFT: " << avg_m_radix_4 << "\n";
    cout << "Radix Split FFT:     " << avg_radix_split << "\n\n";
    cout << "Average sparsity:     " << avg_sparsity << "\n\n";
    cout << "Best algorithm distribution:\n";
    for (const auto& [algo, count] : best_algo_count) {
        cout << algo << ": " << count << " times (" 
             << fixed << setprecision(1) << (100.0 * count / n) << "%)\n";
    }
    
    return 0;
}
