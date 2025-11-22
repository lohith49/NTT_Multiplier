#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;
using cd = complex<double>;
const double PI = acos(-1);


int nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int nextPowerOf4(int n) {
    int p = 1;
    while (p < n) p <<= 2;  
    return p;
}

int distToNextPowerOf2(int n) {
    if (n <= 0) return 0;
    int nextPow2 = nextPowerOf2(n);
    return nextPow2 - n;
}

static bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
static bool is_power_of_4(int n) {
    if (n <= 0) return false;
    if (!is_power_of_2(n)) return false;
    while (n % 4 == 0) n /= 4;
    return n == 1;
}

void fft_radix_2(vector<cd>& A,vector<cd>& X){
    int n=A.size();
    if(n==1){
        return;
    }
    vector<cd>A_even(n/2);
    vector<cd>A_odd(n/2);
    for(int i=0;i<(n/2);i++){
        A_even[i]=A[2*i];
        A_odd[i]=A[2*i+1];
    }
    vector<cd>X_2(n);
    for(int i=0;i<n;i++){
        X_2[i]=X[i]*X[i];
    }
    fft_radix_2(A_even,X_2);
    fft_radix_2(A_odd,X_2);

    for(int i=0;i<(n/2);i++){
        A[i]=A_even[i]+X[i]*A_odd[i];
    }
    for(int i=0;i<(n/2);i++){
        A[i+(n/2)]=A_even[i]+X[i+(n/2)]*A_odd[i];
    }
}


void fft_radix_split(vector<cd> &A,vector<cd>& X) {
    int n = A.size();
    if (n <= 1) return;
    if (n == 2) {
        cd a = A[0];
        cd b = A[1];
        A[0] = a + b;
        A[1] = a - b;
        return;
    }
    vector<cd> A0(n/2), A1(n/4), A3(n/4);
    for (int i = 0; i < n/2; i++){
        A0[i] = A[2 * i];
    }
    for (int i = 0; i < n/4; i++){
        A1[i] = A[4 * i + 1];
        A3[i] = A[4 * i + 3];
    }

    vector<cd> X2(n), X4(n);
    for(int i=0;i<n;i++){
        X2[i] = X[i]*X[i];
        X4[i] = X2[i]*X2[i];
    }
    
    fft_radix_split(A0,X2);
    fft_radix_split(A1,X4);
    fft_radix_split(A3,X4);

    for(int i=0;i<(n/4);i++){
        A[i]=A0[i]+X[i]*A1[i]+X[(3*i)%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[i+(n/4)]=A0[i+(n/4)]+X[i+n/4]*A1[i]+X[(3*(i+n/4))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+(n/2))]=A0[i]+X[i+n/2]*A1[i]+X[(3*(i+n/2))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+3*(n/4))]=A0[i+(n/4)]+X[i+3*n/4]*A1[i]+X[(3*(i+3*n/4))%n]*A3[i];
    }
}




void fft_radix_4(vector<cd> &A,vector<cd>& X) {
    int n = A.size();
    if (n == 1) return;

    vector<cd> A0(n/4), A1(n/4), A2(n/4), A3(n/4);
    for(int i=0;i<n/4;i++){
        A0[i]=A[4*i];
        A1[i]=A[4*i+1];
        A2[i]=A[4*i+2];
        A3[i]=A[4*i+3];
    }
    vector<cd> X4(n);
    for(int i=0;i<n;i++){
        X4[i] = X[i]*X[i]*X[i]*X[i];
    }
    fft_radix_4(A0,X4);
    fft_radix_4(A1,X4);
    fft_radix_4(A2,X4);
    fft_radix_4(A3,X4);

    for(int i=0;i<(n/4);i++){
        A[i]=A0[i]+X[i]*A1[i]+X[(2*i)%n]*A2[i]+X[(3*i)%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[i+(n/4)]=A0[i]+X[i+n/4]*A1[i]+X[(2*(i+n/4))%n]*A2[i]+X[(3*(i+n/4))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+(n/2))]=A0[i]+X[i+2*(n/4)]*A1[i]+X[(2*(i+2*(n/4)))%n]*A2[i]+X[(3*(i+2*(n/4)))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+3*(n/4))]=A0[i]+X[i+3*(n/4)]*A1[i]+X[(2*(i+3*(n/4)))%n]*A2[i]+X[(3*(i+3*(n/4)))%n]*A3[i];
    }
}




void fft_modified_radix_4(vector<cd> &A,vector<cd>& X) {
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
    for(int i=0;i<n/4;i++){
        A0[i]=A[4*i];
        A1[i]=A[4*i+1];
        A2[i]=A[4*i+2];
        A3[i]=A[4*i+3];
    }
    vector<cd> X4(n);
    for(int i=0;i<n;i++){
        X4[i] = X[i]*X[i]*X[i]*X[i];
    }
    fft_modified_radix_4(A0,X4);
    fft_modified_radix_4(A1,X4);
    fft_modified_radix_4(A2,X4);
    fft_modified_radix_4(A3,X4);

    for(int i=0;i<(n/4);i++){
        A[i]=A0[i]+X[i]*A1[i]+X[(2*i)%n]*A2[i]+X[(3*i)%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[i+(n/4)]=A0[i]+X[i+n/4]*A1[i]+X[(2*(i+n/4))%n]*A2[i]+X[(3*(i+n/4))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+(n/2))]=A0[i]+X[i+2*(n/4)]*A1[i]+X[(2*(i+2*(n/4)))%n]*A2[i]+X[(3*(i+2*(n/4)))%n]*A3[i];
    }

    for(int i=0;i<(n/4);i++){
        A[(i+3*(n/4))]=A0[i]+X[i+3*(n/4)]*A1[i]+X[(2*(i+3*(n/4)))%n]*A2[i]+X[(3*(i+3*(n/4)))%n]*A3[i];
    }
}

static string normalize_algo(string s) {
    for (auto &c : s) c = std::tolower(c);
    return s;
}

static bool requires_power_of_4(const string &algoNorm) {
    return (algoNorm == "radix-4" || algoNorm == "modified-radix-4");
}

static void apply_fft(vector<cd> &A, vector<cd> &X, const string &algo) {
    string a = normalize_algo(algo);
    if (a == "radix-2") {
        fft_radix_2(A, X);
    } else if (a == "split-radix" || a == "radix-split") { // accept both label styles
        fft_radix_split(A, X);
    } else if (a == "radix-4") {
        fft_radix_4(A, X);
    } else if (a == "modified-radix-4") {
        fft_modified_radix_4(A, X);
    } else {
        cerr << "ERROR: Unknown FFT algorithm: " << algo << endl;
        exit(1);
    }
}

static pair<string, double> run_ml_predictor(int polynomial_size, double sparsity, int dist_to_next_pow2, int is_power_2, int is_power_4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss << setprecision(6);
    oss << "python3 \"/home/lohith/Desktop/NTT_Multiplier/ML_Model/predict_algorithm.py\""
        << " --polynomial_size " << polynomial_size
        << " --sparsity " << sparsity
        << " --dist_to_next_pow2 " << dist_to_next_pow2
        << " --is_power_2 " << is_power_2
        << " --is_power_4 " << is_power_4
        << " 2>/dev/null"; // silence warnings to simplify parsing

    string cmd = oss.str();
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        cerr << "ERROR: Failed to execute ML predictor command" << endl;
        exit(1);
    }
    
    string output;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int exit_code = pclose(pipe);
    
    if (exit_code != 0) {
        cerr << "ERROR: ML predictor script failed with exit code " << exit_code << endl;
        exit(1);
    }

    string algo;
    double confidence = -1.0; // invalid default to detect parsing failure
    std::istringstream in(output);
    string line;
    
    while (std::getline(in, line)) {
        auto pos = line.find("Best Algorithm:");
        if (pos == string::npos) pos = line.find("Predicted Algorithm:");
        if (pos != string::npos) {
            string val = line.substr(pos + string("Best Algorithm:").size());
            auto colon = val.find(":");
            if (colon != string::npos) val = val.substr(colon+1);
            auto l = val.find_first_not_of(" \t\xF0\x9F\x8E\xAF\xF0\x9F\x93\x8A");
            auto r = val.find_last_not_of(" \t\r\n");
            if (l != string::npos) {
                algo = val.substr(l, r == string::npos ? string::npos : r - l + 1);
            }
        }
        
        pos = line.find("Confidence:");
        if (pos != string::npos) {
            string val = line.substr(pos + string("Confidence:").size());
            auto colon = val.find(":");
            if (colon != string::npos) val = val.substr(colon+1);
            auto l = val.find_first_not_of(" \t\xF0\x9F\x93\x8A");
            if (l != string::npos) {
                string conf_str = val.substr(l);
                try {
                    confidence = std::stod(conf_str);
                } catch (...) {
                    cerr << "ERROR: Failed to parse confidence value: " << conf_str << endl;
                    exit(1);
                }
            }
        }
    }
    
    if (algo.empty()) {
        cerr << "ERROR: Could not parse algorithm from ML predictor output" << endl;
        cerr << "Output was: " << output << endl;
        exit(1);
    }
    
    if (confidence < 0.0) {
        cerr << "ERROR: Could not parse confidence score from ML predictor output" << endl;
        cerr << "Output was: " << output << endl;
        exit(1);
    }
    
    for (auto &c : algo) c = std::tolower(c);
    
    if (algo != "radix-2" && algo != "radix-4" && algo != "modified-radix-4" && 
        algo != "radix-split" && algo != "split-radix") {
        cerr << "ERROR: Unknown algorithm predicted: " << algo << endl;
        exit(1);
    }
    
    return make_pair(algo, confidence);
}

void parallel_multiply(vector<cd>& fa, const vector<cd>& fb) {
    int n = static_cast<int>(fa.size());
    for (int i = 0; i < n; ++i) {
        fa[i] *= fb[i];
    }
}


vector<cd> multiply_poly(vector<int>& A,vector<int>& B, const string &algoA, const string &algoB){
    string aA = normalize_algo(algoA);
    string aB = normalize_algo(algoB);
    int needed = (int)A.size() + (int)B.size();
    
    int n = nextPowerOf2(needed);
    
    vector<cd> X(n), X_bar(n);
    double ang = 2*PI/n;
    for(int i=0;i<n;i++){
        X[i] = cd(cos(ang*i), sin(ang*i));
        X_bar[i] = cd(cos(-ang*i), sin(-ang*i));
    }
    vector<cd> fa(n, 0), fb(n, 0);
    for (int i = 0; i < (int)A.size(); i++) fa[i] = A[i];
    for (int i = 0; i < (int)B.size(); i++) fb[i] = B[i];

    apply_fft(fa, X, aA);
    apply_fft(fb, X, aB);

    parallel_multiply(fa, fb);

    fft_radix_2(fa, X_bar);
    
    for (int i = 0; i < n; i++) {
        fa[i] /= n;
    }

    return fa;
}




int main() {
    vector<int>polyAsize ={
        2, 9, 696, 1058, 1169, 1760, 3383, 917, 1189, 4630,
        1462, 4276, 2376, 3931, 851, 3027, 4999, 4469, 1728, 2277,
        2910, 767, 938, 2330, 4122, 3456, 1866, 2432, 4346, 4019,
        4867, 2649, 2774, 4914, 2870, 3638, 4672, 4401, 4590, 3164,
        3724, 76, 1231, 2108, 249, 4143, 2662, 110, 4803, 4825,
        1278, 1948, 865, 1627, 1679, 557, 2121, 4088, 2073, 1375,
        2838, 4571, 3407, 3256, 1925, 9, 3554, 3258, 2750, 3108,
        1847, 4455, 474, 2480, 3968, 589, 2531, 1604, 3811, 1475,
        343, 262, 507, 1333, 2242, 8, 4806, 3131, 1548, 2922,
        1080, 3417, 3738, 3899, 151, 3600, 731, 2134, 2591, 399

    };
    
    vector<int>polyBsize= {
        2, 4, 4821, 958, 2872, 3957, 729, 1787, 140, 553,
        154, 2663, 3450, 3667, 1673, 653, 1530, 4710, 3839, 3848,
        2142, 2602, 4057, 3382, 108, 3728, 2213, 3704, 2126, 832,
        446, 4749, 2997, 1327, 2019, 3292, 3208, 4383, 3568, 4486,
        1993, 4961, 3091, 2383, 3466, 837, 4265, 372, 1395, 4839,
        4682, 2940, 3810, 2474, 2052, 615, 2645, 3964, 1659, 4311,
        280, 1284, 1850, 60, 3139, 2580, 734, 1052, 4724, 2425,
        3697, 4214, 4574, 4018, 4508, 1201, 4538, 4164, 1820, 592,
        4183, 1775, 2794, 2369, 3251, 510, 2865, 981, 2278, 3539,
        2712, 1239, 3067, 377, 4942, 1937, 1520, 1418, 277, 1139

    };

    ofstream csv("ml_predictions_updated.csv");
    if (!csv.is_open()) {
        cerr << "ERROR: Unable to open ml_predictions_updated.csv for writing" << endl;
        return 1;
    }
    csv << "Polynomial A size,Polynomial B size,Predicted A-algo,Predicted B-algo,"
        << "\"(radix-2,radix-2)time\",\"(split-radix,split-radix)time\",\"(modified-radix-4,modified-radix-4)time\"," 
        << "\"(radix-2,split-radix)time\",\"(radix-2,modified-radix-4)time\",\"(split-radix,radix-2)time\"," 
        << "\"(split-radix,modified-radix-4)time\",\"(modified-radix-4,radix-2)time\",\"(modified-radix-4,split-radix)time\"\n";

    size_t ncases = min(polyAsize.size(), polyBsize.size());
    if (ncases == 0) {
        cerr << "ERROR: Empty size vectors" << endl;
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100, 100);

    const vector<pair<string,string>> pairOrder = {
        {"radix-2","radix-2"},
        {"split-radix","split-radix"},
        {"modified-radix-4","modified-radix-4"},
        {"radix-2","split-radix"},
        {"radix-2","modified-radix-4"},
        {"split-radix","radix-2"},
        {"split-radix","modified-radix-4"},
        {"modified-radix-4","radix-2"},
        {"modified-radix-4","split-radix"}
    };

    for (size_t idx = 0; idx < ncases; ++idx) {
        int x = polyAsize[idx];
        int y = polyBsize[idx];
        if (x <= 0 || y <= 0) {
            cerr << "ERROR: Invalid polynomial sizes at index " << idx << endl;
            continue;
        }

        vector<int> a(x), b(y);
        int zerosAinOrig = 0, zerosBinOrig = 0;
        for (int i = 0; i < x; ++i) { a[i] = dist(gen); if (a[i] == 0) ++zerosAinOrig; }
        for (int j = 0; j < y; ++j) { b[j] = dist(gen); if (b[j] == 0) ++zerosBinOrig; }

        int paddedLenA = x + y;  // Transform length needed for A
        long long zerosA = (long long)zerosAinOrig + (paddedLenA - x);
        double sparsityA = (double)zerosA / (double)paddedLenA;
        int distToPow2A = distToNextPowerOf2(paddedLenA);
        int p2flagA = is_power_of_2(paddedLenA) ? 1 : 0;
        int p4flagA = is_power_of_4(paddedLenA) ? 1 : 0;

        int paddedLenB = x + y;  // Same transform length for B
        long long zerosB = (long long)zerosBinOrig + (paddedLenB - y);
        double sparsityB = (double)zerosB / (double)paddedLenB;
        int distToPow2B = distToNextPowerOf2(paddedLenB);
        int p2flagB = is_power_of_2(paddedLenB) ? 1 : 0;
        int p4flagB = is_power_of_4(paddedLenB) ? 1 : 0;

        auto [predA, confA] = run_ml_predictor(paddedLenA, sparsityA, distToPow2A, p2flagA, p4flagA);
        auto [predB, confB] = run_ml_predictor(paddedLenB, sparsityB, distToPow2B, p2flagB, p4flagB);

        string predA_norm = normalize_algo(predA);
        string predB_norm = normalize_algo(predB);
        
        if (predA_norm == "radix-4") predA_norm = string("modified-radix-4");
        if (predB_norm == "radix-4") predB_norm = string("modified-radix-4");  
        if (predA_norm == "radix-split") predA_norm = string("split-radix");
        if (predB_norm == "radix-split") predB_norm = string("split-radix");

        vector<long long> timesMs(pairOrder.size());
        for (size_t pi = 0; pi < pairOrder.size(); ++pi) {
            auto start = high_resolution_clock::now();
            vector<cd> Result = multiply_poly(a, b, pairOrder[pi].first, pairOrder[pi].second);
            auto stop = high_resolution_clock::now();
            timesMs[pi] = duration_cast<milliseconds>(stop - start).count();
        }

        size_t predIdx = pairOrder.size();
        for (size_t i = 0; i < pairOrder.size(); ++i) {
            if (pairOrder[i].first == predA_norm && pairOrder[i].second == predB_norm) { predIdx = i; break; }
        }
        if (predIdx == pairOrder.size()) {
            cerr << "ERROR: Predicted pair not in allowed set for index " << idx << endl;
            return 1;
        }

        if(predA=="radix-split")predA="split-radix";
         if(predB=="radix-split")predB="split-radix";
        csv << x << "," << y << ","
            << predA << ","
            << predB << ",";
        for (size_t i = 0; i < pairOrder.size(); ++i) {
            csv << timesMs[i];
            csv << (i + 1 == pairOrder.size() ? '\n' : ',');
        }

        cout << "Case " << (idx + 1) << ": A=" << x << ", B=" << y
             << ", predA=" << predA_norm << ", predB=" << predB_norm
             << ", pred_ms=" << timesMs[predIdx] 
             << ", distToPow2=" << distToPow2A << "\n";
    }

    csv.close();
    return 0;
}