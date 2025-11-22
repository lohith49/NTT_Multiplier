## FFT ML Pipeline Quick Run

Prereqs (first time only):
```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost joblib
```

1. Generate baseline FFT performance data (creates fft_performance_results.csv)
```bash
g++ -O2 -std=c++17 generate_dataset.cpp -o generate_dataset
./generate_dataset
```

2. Create synthetic extra rows
```bash
python3 ML_Model/genetic.py
```

3. Verify synthetic rows via C++ timing (outputs verify_extra_data.csv)
```bash
g++ -O2 -std=c++17 verify.cpp -o verify
./verify
```

4. Merge verified extras into clean dataset (fft_clean_extradata.csv)
```bash
python3 ML_Model/merge_extradata.py
```

5. Train models and save artifacts (XGBoost + meta info)
```bash
python3 ML_Model/main.py
```

6. Produce prediction table for random polynomials
```bash
g++ -O2 -std=c++17 ML_Pipeline_1.cpp -o ml
./ml
```

Files of interest (after steps):
- ML_Model/fft_performance_results.csv
- ML_Model/extra_data.csv
- ML_Model/verify_extra_data.csv
- ML_Model/fft_clean_extradata.csv
- ML_Model/xgboost_fft_model.json
- ml (executable) output CSV
