# Rossmann Store Sales Forecasting

End-to-end machine learning pipeline that predicts daily sales for Rossmann drug store chain outlets, using historical sales, store metadata, promotions, and holiday information. The project covers exploratory data analysis, feature engineering, model training/tuning, and model explainability with SHAP.

## Project Structure

```
├── notebooks/
│   ├── eda.ipynb                     # Exploratory data analysis & business insights
│   ├── feature_engineering.ipynb     # Data cleaning, feature creation, encoding
│   ├── model.ipynb                   # Model training, tuning, and evaluation
│   └── shap_explainability.ipynb     # SHAP-based model interpretability
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── store.csv
│   └── processed_train.csv           # Output of feature engineering
└── models/
    └── xgboost_model.pkl             # Final tuned XGBoost model
```

## Workflow

**1. Exploratory Data Analysis (`eda.ipynb`)**
- Analyzed 1M+ sales records across 1,115 stores
- Identified weekly/yearly seasonality, promo lift, and store-type sales patterns
- Confirmed closed stores report zero sales, and missing values are structural (tied to store characteristics) rather than random

**2. Feature Engineering (`feature_engineering.ipynb`)**
- Merged sales data with store metadata
- Imputed missing values (competition distance, promo2 timing, etc.)
- Engineered date-based features: year, month, day, week of year, day of week, quarter, weekend/month-start/month-end flags
- Derived competition and promo "age" features (months since competition/promo began)
- Label-encoded categorical variables (store type, assortment, promo interval, state holiday)
- Filtered to open stores with positive sales only

**3. Modeling (`model.ipynb`)**
- Time-based train/validation/test split (validation: Jun–Aug 2015)
- Benchmarked Linear Regression, XGBoost, and LightGBM
- Tuned XGBoost via `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation
- Saved the final model with `joblib`

**4. Explainability (`shap_explainability.ipynb`)**
- Used SHAP to interpret the tuned XGBoost model
- Identified top sales drivers: Promo, CompetitionDistance, Assortment, Store
- Investigated best/worst individual predictions and the largest prediction errors
- Compared SHAP importances against native XGBoost feature importances

## Results

| Model             | MAE     | RMSE    | R²     |
|-------------------|---------|---------|--------|
| Linear Regression | 1986.49 | 2698.86 | 0.237  |
| LightGBM          | 1163.93 | 1576.26 | 0.740  |
| XGBoost (baseline)| 869.90  | 1192.31 | 0.851  |
| **XGBoost (tuned)** | **785.88**  | **1099.94**  | **0.873**  |

The tuned XGBoost model was selected as the final model, explaining ~87% of the variance in daily sales on the validation set.

## Key Insights

- Promotions are the strongest driver of sales, followed by competition distance and store assortment type
- Sales exhibit clear weekly and yearly seasonality
- Store type and assortment strongly influence baseline sales levels
- Model errors are concentrated in high-sales outlier days, suggesting room for store-specific or event-based features in future iterations

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Matplotlib` · `Plotly`

## Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib plotly joblib
```

Run the notebooks in order: `eda.ipynb` → `feature_engineering.ipynb` → `model.ipynb` → `shap_explainability.ipynb`

## Future Improvements

- Incorporate lag/rolling-window sales features
- Experiment with store-level embeddings or hierarchical models
- Deploy the model behind a lightweight API for real-time inference
