#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using cd = std::complex<double>;
static constexpr double PI = 3.141592653589793238462643383279502884;

static inline int nextPowerOf2(int n) { int p = 1; while (p < n) p <<= 1; return p; }
static inline bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }
static inline std::string trim(const std::string &s){ size_t l = s.find_first_not_of(" \t\r\n"); if (l==std::string::npos) return ""; size_t r = s.find_last_not_of(" \t\r\n"); return s.substr(l, r-l+1); }

static void fft_radix_2(std::vector<cd>& A, std::vector<cd>& X){
	int n = (int)A.size();
	if (n == 1) return;
	std::vector<cd> A_even(n/2), A_odd(n/2);
	for (int i = 0; i < n/2; ++i) { A_even[i] = A[2*i]; A_odd[i] = A[2*i+1]; }
	std::vector<cd> X2(n); for (int i = 0; i < n; ++i) X2[i] = X[i] * X[i];
	fft_radix_2(A_even, X2);
	fft_radix_2(A_odd,  X2);
	for (int i = 0; i < n/2; ++i) A[i] = A_even[i] + X[i] * A_odd[i];
	for (int i = 0; i < n/2; ++i) A[i + n/2] = A_even[i] + X[i + n/2] * A_odd[i];
}

static void fft_radix_split(std::vector<cd>& A, std::vector<cd>& X){
	int n = (int)A.size();
	if (n <= 1) return;
	if (n == 2) { cd a=A[0], b=A[1]; A[0]=a+b; A[1]=a-b; return; }
	std::vector<cd> A0(n/2), A1(n/4), A3(n/4);
	for (int i=0;i<n/2;++i) A0[i]=A[2*i];
	for (int i=0;i<n/4;++i){ A1[i]=A[4*i+1]; A3[i]=A[4*i+3]; }
	std::vector<cd> X2(n), X4(n);
	for (int i=0;i<n;++i){ X2[i]=X[i]*X[i]; X4[i]=X2[i]*X2[i]; }
	fft_radix_split(A0,X2); fft_radix_split(A1,X4); fft_radix_split(A3,X4);
	for (int i=0;i<n/4;++i) A[i]         = A0[i]           + X[i]           * A1[i] + X[(3*i)%n]           * A3[i];
	for (int i=0;i<n/4;++i) A[i + n/4]   = A0[i + n/4]     + X[i + n/4]     * A1[i] + X[(3*(i+n/4))%n]    * A3[i];
	for (int i=0;i<n/4;++i) A[i + n/2]   = A0[i]           + X[i + n/2]     * A1[i] + X[(3*(i+n/2))%n]    * A3[i];
	for (int i=0;i<n/4;++i) A[i + 3*n/4] = A0[i + n/4]     + X[i + 3*n/4]   * A1[i] + X[(3*(i+3*n/4))%n]  * A3[i];
}

static void fft_modified_radix_4(std::vector<cd>& A, std::vector<cd>& X){
	int n = (int)A.size();
	if (n == 1) return;
	if (n == 2) { cd a=A[0], b=A[1]; A[0]=a+b; A[1]=a-b; return; }
	std::vector<cd> A0(n/4), A1(n/4), A2(n/4), A3(n/4);
	for (int i=0;i<n/4;++i){ A0[i]=A[4*i]; A1[i]=A[4*i+1]; A2[i]=A[4*i+2]; A3[i]=A[4*i+3]; }
	std::vector<cd> X4(n); for (int i=0;i<n;++i) X4[i]=X[i]*X[i]*X[i]*X[i];
	fft_modified_radix_4(A0,X4); fft_modified_radix_4(A1,X4); fft_modified_radix_4(A2,X4); fft_modified_radix_4(A3,X4);
	for (int i=0;i<n/4;++i) A[i]         = A0[i]           + X[i]           * A1[i] + X[(2*i)%n]          * A2[i] + X[(3*i)%n]          * A3[i];
	for (int i=0;i<n/4;++i) A[i + n/4]   = A0[i]           + X[i + n/4]     * A1[i] + X[(2*(i+n/4))%n]    * A2[i] + X[(3*(i+n/4))%n]    * A3[i];
	for (int i=0;i<n/4;++i) A[i + n/2]   = A0[i]           + X[i + n/2]     * A1[i] + X[(2*(i+n/2))%n]    * A2[i] + X[(3*(i+n/2))%n]    * A3[i];
	for (int i=0;i<n/4;++i) A[i + 3*n/4] = A0[i]           + X[i + 3*n/4]   * A1[i] + X[(2*(i+3*n/4))%n]  * A2[i] + X[(3*(i+3*n/4))%n]  * A3[i];
}

struct Row { int polySize; double sparsity; int distNextPow2; int isPow2; int isPow4; std::string bestAlgo; };

static bool parse_row(const std::string &line, Row &row){
	std::vector<std::string> parts; parts.reserve(6);
	std::string cur; std::stringstream ss(line);
	while (std::getline(ss, cur, ',')) parts.push_back(trim(cur));
	if (parts.size() < 6) return false;
	try{
		row.polySize = std::stoi(parts[0]);
		row.sparsity = std::stod(parts[1]);
		row.distNextPow2 = std::stoi(parts[2]);
		row.isPow2 = std::stoi(parts[3]);
		row.isPow4 = std::stoi(parts[4]);
		row.bestAlgo = parts[5];
	}catch(...){ return false; }
	return true;
}


int main(int argc, char** argv){
	// Default paths: use ML_Model directory next to this executable's source location
	std::string inPath = "ML_Model/extra_data.csv";
	std::string outPath = "ML_Model/verify_extra_data.csv";

	size_t max_rows = 0; // 0 = all
	for (int i=1;i<argc;++i){
		std::string a=argv[i];
		if (a=="--max-rows" && i+1<argc){ max_rows = (size_t)std::stoull(argv[++i]); }
		else if ((a=="--in" || a=="-i") && i+1<argc){ inPath = argv[++i]; }
		else if ((a=="--out" || a=="-o") && i+1<argc){ outPath = argv[++i]; }
	}

	std::ifstream fin(inPath);
	if (!fin.is_open()) { std::cerr << "ERROR: Cannot open input: " << inPath << "\n"; return 1; }
	std::ofstream fout(outPath);
	if (!fout.is_open()) { std::cerr << "ERROR: Cannot open output: " << outPath << "\n"; return 1; }

	std::string header; std::getline(fin, header);
	if (header.find("Polynomial_Size") == std::string::npos) { std::cerr << "ERROR: Unexpected header in extra_data.csv\n"; return 1; }
	fout << header << ",Actual_Best_Algorithm\n";

	std::mt19937 rng(42);
	std::uniform_int_distribution<int> valdist(-100, 100);

	size_t rows = 0, matched = 0, processed = 0;
	std::string line;
	while (std::getline(fin, line)){
		if (max_rows && processed >= max_rows) break;
		line = trim(line); if (line.empty()) continue;
		Row r{}; if (!parse_row(line, r)) { std::cerr << "WARN: Skipping malformed line\n"; continue; }

		long long n_ll = (long long)r.polySize + (long long)r.distNextPow2;
		if (n_ll <= 0 || n_ll > INT_MAX) { std::cerr << "WARN: Invalid n computed, skipping\n"; continue; }
		int n = (int)n_ll; if (!is_power_of_2(n)) n = nextPowerOf2(r.polySize); if (n < 2) n = 2;

		std::vector<cd> A(n, cd(0,0));
		int zerosTarget = (int)llround(r.sparsity * (double)n); zerosTarget = std::max(0, std::min(zerosTarget, n));
		int nonZeros = n - zerosTarget;
		std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0); std::shuffle(idx.begin(), idx.end(), rng);
		for (int i = 0; i < nonZeros; ++i) { int pos = idx[i]; int v = valdist(rng); if (v == 0) v = 1; A[pos] = cd((double)v, 0.0); }

		std::vector<cd> X(n); double ang = 2*PI/n; for (int i=0;i<n;++i) X[i] = cd(std::cos(ang*i), std::sin(ang*i));

		auto time_algo = [&](int algo)->long long{
			int reps = (n <= 1024 ? 5 : 1); long long total_us = 0;
			for (int k=0;k<reps;++k){ std::vector<cd> B=A; auto t0=std::chrono::high_resolution_clock::now();
				if (algo==0) fft_radix_2(B,X); else if (algo==1) fft_radix_split(B,X); else fft_modified_radix_4(B,X);
				auto t1=std::chrono::high_resolution_clock::now(); total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count(); }
			return total_us / (long long)reps; };

		long long t0 = time_algo(0); // r2
		long long t1 = time_algo(1); // split
		long long t2 = time_algo(2); // mr4
		std::string actual = (t0<=t1 && t0<=t2) ? "Radix-2" : (t1<=t0 && t1<=t2) ? "Radix-Split" : "Modified-Radix-4";

		// Compare (case-insensitive, normalize split)
		auto tolower_str = [](std::string s){ for (auto &c: s) c = (char)std::tolower((unsigned char)c); return s; };
		std::string provided = tolower_str(r.bestAlgo);
		std::string actualNorm = tolower_str(actual);
		if (provided=="radix-split") provided="split-radix"; if (actualNorm=="radix-split") actualNorm="split-radix";
		if (provided==actualNorm) matched++; rows++; processed++;

		// Write out row with Actual_Best_Algorithm
		fout << r.polySize << "," << std::setprecision(10) << r.sparsity << "," << r.distNextPow2
			 << "," << r.isPow2 << "," << r.isPow4 << "," << r.bestAlgo << "," << actual << "\n";
	}

	fin.close(); fout.close();
	std::cout << "Verification complete. Rows: " << rows << ", Matches: " << matched
			  << ", Match%: " << (rows ? (100.0*matched/rows) : 0.0) << "%\n";
	std::cout << "Output written: " << outPath << "\n";
	return 0;
}
