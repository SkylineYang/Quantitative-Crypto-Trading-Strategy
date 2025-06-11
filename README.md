# Project 3: Crypto Algorithm Trading

### 1. How to run our code?

Our main work is located in `Project3_Stage1&2.ipynb`, so please refer to this specific file.

In this file, first 1-3 parts are our research and design, the 4th part is the backtest (which we highly recommand you to directly run).

### 2. Our project files structure

```plaintext
.
│  DataPreprocessing.py
│  ManualScalar.py
│  NeuralNetworks.py
│  Pro3_Stage1_SP25.xlsx
│  Project3_Stage1.ipynb
│  README.md
│
├─models
│      best_gru_model_config.json
│      best_lstm_model_config.json
│      GRUmodel.pth
│      LSTMmodel.pth
│      RandomForest.joblib
│      XGBoost.joblib
```

DataPreprocessing.py: Seal all data preprocessing related functions inside

ManualScalar.py: Implement manual MinMaxScalar

NeuralNetworks.py: Define LSTM and GRU
