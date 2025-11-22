sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-pip

cd /home/lohith/Desktop/NTT_Multiplier
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost joblib

python3 ML_Model/genetic.py

g++ -O2 -std=c++17 verify.cpp -o verify
./verify --in ML_Model/extra_data.csv --out ML_Model/verify_extra_data.csv

python3 ML_Model/merge_extradata.py

python3 ML_Model/main.py

g++ -O2 -std=c++17 ML_Pipeline_1.cpp -o ml_pipeline
./ml_pipeline
