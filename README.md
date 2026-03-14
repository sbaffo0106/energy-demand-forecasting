# Electricity Demand Forecasting with Machine Learning

This project builds a machine learning pipeline to forecast **hourly electricity demand** using historical load data.

The goal is to explore how different regression models perform on a time-series forecasting problem and to identify the most informative features driving electricity demand.

---

## Project overview

Electricity demand forecasting is a critical task for:

- energy grid management
- operational planning
- demand response strategies
- energy market optimization

This project implements a full machine learning workflow for predicting hourly electricity load using historical observations and engineered temporal features.

---

## Pipeline

The project follows a structured ML workflow:

1. Load and preprocess the dataset  
2. Validate timestamps and data consistency  
3. Create calendar, cyclical, lag, and rolling features  
4. Train multiple regression models  
5. Evaluate them with a chronological train/test split  
6. Validate model robustness with time-series cross-validation 

Pipeline overview:

Raw data  
↓  
Preprocessing  
↓  
Feature engineering  
↓  
Train/Test split  
↓  
Model training  
↓  
Evaluation  

The full implementation of this pipeline can be explored in the accompanying notebook `energy_demand_forecasting.ipynb`.

---

## Dataset

The dataset contains **hourly electricity demand measurements**.

Main variable:

- **PJME_MW** → hourly electricity load (target variable)

The dataset is indexed by timestamp and sorted chronologically to preserve the time-series structure.

---

## Feature engineering

Several types of features were created to capture temporal patterns.

### Calendar features
- hour
- day of week
- month
- weekend indicator

### Cyclical encoding
To properly represent cyclic variables:

- hour_sin / hour_cos
- dow_sin / dow_cos

### Lag features
Past demand values used as predictors:

- lag_1
- lag_24
- lag_168

### Rolling statistics
Past demand variability:

- rolling_mean_24
- rolling_std_24

All rolling features are computed **only from past observations** to avoid target leakage.

---

## Models compared

Three regression models were evaluated:

- **Ridge Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Models were trained on historical data and evaluated using a **chronological train/test split**.

---

## Evaluation metrics

Model performance was evaluated using:

- **RMSE** – Root Mean Squared Error
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of determination

These metrics provide complementary views on prediction accuracy.

---

## Results

Among the tested models, **Random Forest achieved the best predictive performance**, significantly outperforming the linear baseline.

Key observations:

- Electricity demand shows strong **short-term persistence**
- **Lag features** are the most informative predictors
- Ensemble models capture nonlinear patterns better than linear regression

---

## Time-series validation

To ensure model robustness, the models were also evaluated using **TimeSeriesSplit cross-validation**.

This approach preserves chronological ordering and prevents information leakage between training and validation sets.

---

## Repository structure

```
energy-demand-forecasting/
│
├── data/
│   └── PJME_hourly.csv
│
├── preprocessing.py
├── features.py
├── train.py
├── evaluate.py
│
├── energy_demand_forecasting.ipynb
│
├── README.md
└── requirements.txt
```

---

## Technologies used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## How to run the project

1. **Clone the repository**

```bash
git clone https://github.com/sbaffo0106/energy-demand-forecasting.git
cd energy-demand-forecasting
```

2. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

3. **Open the notebook and run the cells sequentially**

Open the file `energy_demand_forecasting.ipynb` and execute the cells from top to bottom.

---

## Future improvements

Possible extensions of this project include:

- hyperparameter tuning
- advanced boosting models (XGBoost / LightGBM)
- probabilistic forecasting
- multi-step forecasting
- deployment as a prediction API

---

## Author

**Antonio Sbaffoni**

Machine Learning project focused on **time-series forecasting techniques applied to energy demand prediction**.

GitHub: https://github.com/sbaffo0106  
LinkedIn: https://www.linkedin.com/in/dr-antonio-sbaffoni-85644a184/
