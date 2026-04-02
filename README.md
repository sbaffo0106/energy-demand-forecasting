# Day-Ahead Electricity Demand Forecasting with Machine Learning and Weather Features

This project initially built a machine learning pipeline to forecast **hourly electricity demand** using historical load data. The goal was to explore how different regression models perform on a time-series forecasting problem and to identify the most informative features driving electricity demand.

## Project upgrade

This project extends a previous short-term forecasting prototype by addressing key real-world limitations:

- shifting from +1h prediction to **day-ahead (+24h) forecasting**
- incorporating **weather features (temperature)** as exogenous variables
- benchmarking machine learning models against **statistical baselines**
- improving evaluation under realistic forecasting constraints

The goal is to move from a simulation setup to a more **industry-relevant forecasting pipeline**.

---

## Project overview

Electricity demand forecasting is a critical task for:

- energy grid management
- operational planning
- demand response strategies
- energy market optimization

This project implements a full machine learning workflow for predicting hourly electricity load using historical observations and engineered temporal features.

The updated version focuses on **day-ahead forecasting (+24h horizon)**, which aligns with real-world energy market requirements.

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

The improved pipeline can be explored in the accompanying notebooks:

- `notebooks/02_day_ahead_forecasting.ipynb` (updated version)
- `notebooks/01_baseline_exploration.ipynb` (initial prototype)

---

## Dataset

The dataset contains **hourly electricity demand measurements**.

In the updated version of the project, weather data (temperature) is incorporated to better capture external drivers of electricity demand.

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

### Weather features
- temperature
- lagged temperature
- rolling temperature statistics

All rolling features are computed **only from past observations** to avoid target leakage.

---

## Models compared

The following models were evaluated:

### Baselines
- Naive forecast (previous day same hour)

### Statistical model
- **Exponential Smoothing** (Holt-Winters)

### Machine learning models
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

Among the tested models, Random Forest achieved the best performance for short-term prediction, while statistical baselines provided competitive results and better interpretability for trend and seasonality.

Key observations:

- Electricity demand shows strong **short-term persistence**
- **Lag features** are the most informative predictors
- Ensemble models capture nonlinear patterns better than linear regression
- Weather variables (temperature) improve predictive performance, especially during seasonal variations
- Machine learning models perform well in short-term prediction due to strong autocorrelation
- Statistical models provide better interpretability of trend and seasonality

---

## Time-series validation

To ensure model robustness, the models were also evaluated using **TimeSeriesSplit cross-validation**.

This approach preserves chronological ordering and prevents information leakage between training and validation sets.

---

## Real-world considerations

This project highlights several important aspects of electricity demand forecasting:

- Tree-based models (e.g. Random Forest) perform well but struggle with **extrapolation beyond training data**
- Forecast accuracy tends to decrease under **extreme conditions** (e.g. temperature spikes)
- Model performance depends strongly on the **forecasting horizon**

These challenges are critical in real-world applications such as energy trading and grid management.

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
├── notebooks/
│   └── 01_baseline_exploration.ipynb
│   └── 02_day_ahead_forecasting.ipynb
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

3. **Open one of the notebooks and run the cells sequentially:**

- `notebooks/02_day_ahead_forecasting.ipynb` (recommended)
- `notebooks/01_baseline_exploration.ipynb`

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