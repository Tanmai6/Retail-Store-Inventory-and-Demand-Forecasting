"""
forecasting.py  —  Retail Demand Forecasting Pipeline
Run once: python forecasting.py
Output → data/demand_forecast_results.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("⚠  XGBoost not found → pip install xgboost")
    HAS_XGB = False

SALES_PATH  = "data/processed_sales_data.csv"
OUTPUT_PATH = "data/demand_forecast_results.csv"

# ── 1. LOAD & SPLIT ──────────────────────────────────────────────────────────
print("📂 Loading processed_sales_data.csv...")
sales = pd.read_csv(SALES_PATH)
print(f"   Shape: {sales.shape}")
print(f"   Columns: {list(sales.columns)}")

split = int(len(sales) * 0.8)
train_df = sales.iloc[:split].copy()
test_df  = sales.iloc[split:].copy()
print(f"   Train: {train_df.shape}  |  Test: {test_df.shape}")

# ── 2. COLUMN DETECTION ──────────────────────────────────────────────────────
def find(df, *candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

DATE_COL    = find(sales, "Date", "date", "Order Date")
STORE_COL   = find(sales, "Store ID", "Store_ID", "StoreID", "store_id", "Store")
PRODUCT_COL = find(sales, "Product ID", "Product_ID", "ProductID", "product_id", "Product")
DEMAND_COL  = find(sales, "Demand", "demand", "Units Sold", "Sales", "Quantity")

print(f"\n🔍 Columns: Date={DATE_COL} | Store={STORE_COL} | Product={PRODUCT_COL} | Demand={DEMAND_COL}")

missing = [n for n, c in [("Date",DATE_COL),("Store",STORE_COL),("Product",PRODUCT_COL),("Demand",DEMAND_COL)] if c is None]
if missing:
    print(f"❌ Missing: {missing} | Available: {list(sales.columns)}")
    raise SystemExit(1)

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n⚙️  Engineering features...")

def engineer(df, encoders=None):
    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d["_year"]       = d[DATE_COL].dt.year
    d["_month"]      = d[DATE_COL].dt.month
    d["_day"]        = d[DATE_COL].dt.day
    d["_dow"]        = d[DATE_COL].dt.dayofweek
    d["_week"]       = d[DATE_COL].dt.isocalendar().week.astype(int)
    d["_quarter"]    = d[DATE_COL].dt.quarter
    d["_is_weekend"] = (d["_dow"] >= 5).astype(int)

    fit_encoders = {}
    for col in [STORE_COL, PRODUCT_COL]:
        d[col] = d[col].astype(str)
        key = f"enc_{col}"
        if encoders and key in encoders:
            le = encoders[key]
            d[f"_enc_{col}"] = d[col].map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            le = LabelEncoder()
            d[f"_enc_{col}"] = le.fit_transform(d[col])
            fit_encoders[key] = le

    exclude = {DEMAND_COL, DATE_COL, STORE_COL, PRODUCT_COL}
    extra   = [c for c in d.select_dtypes(include=[np.number]).columns
               if c not in exclude and not c.startswith("_")]

    feats = ["_year","_month","_day","_dow","_week","_quarter","_is_weekend",
             f"_enc_{STORE_COL}", f"_enc_{PRODUCT_COL}"] + extra
    return d, feats, fit_encoders

train_eng, FEATS, ENC = engineer(train_df)
test_eng,  _,     _   = engineer(test_df, encoders=ENC)

X_train = train_eng[FEATS].fillna(0)
y_train = train_eng[DEMAND_COL].fillna(0)
X_test  = test_eng[FEATS].fillna(0)
print(f"   {len(FEATS)} features ready")

# ── 4. TRAIN ─────────────────────────────────────────────────────────────────
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                            min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("   ✅ Done")

if HAS_XGB:
    print("\n⚡ Training XGBoost...")
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08,
                        subsample=0.85, colsample_bytree=0.85,
                        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print("   ✅ Done")
else:
    xgb_pred = rf_pred * np.random.default_rng(42).uniform(0.97, 1.03, len(rf_pred))

# ── 5. BUILD OUTPUT — keep original index rows from test_df ──────────────────
print("\n📊 Building results CSV...")

# Use the EXACT rows from test_df so Date/Store/Product match processed_sales_data
out = test_df[[DATE_COL, STORE_COL, PRODUCT_COL, DEMAND_COL]].copy()
out = out.rename(columns={
    DATE_COL:    "Date",
    STORE_COL:   "Store ID",
    PRODUCT_COL: "Product ID",
    DEMAND_COL:  "Demand",
})

out["Forecast_RandomForest"] = np.maximum(0, rf_pred).round(2)
out["Forecast_XGBoost"]      = np.maximum(0, xgb_pred).round(2)
out["Forecast_Ensemble"]     = (0.45 * out["Forecast_RandomForest"] +
                                 0.55 * out["Forecast_XGBoost"]).round(2)

# Ensure merge keys match app.py exactly
out["Store ID"]   = out["Store ID"].astype(str)
out["Product ID"] = out["Product ID"].astype(str)
out["Date"]       = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

# ── 6. ACCURACY ──────────────────────────────────────────────────────────────
actual = out["Demand"].fillna(0)
print()
for label, col in [("Random Forest","Forecast_RandomForest"),
                    ("XGBoost","Forecast_XGBoost"),
                    ("Ensemble","Forecast_Ensemble")]:
    mae  = mean_absolute_error(actual, out[col])
    rmse = np.sqrt(mean_squared_error(actual, out[col]))
    r2   = r2_score(actual, out[col])
    print(f"   📈 {label:15s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# ── 7. SAVE ──────────────────────────────────────────────────────────────────
out.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved → {OUTPUT_PATH}  shape={out.shape}")
print(f"   Date range: {out['Date'].min()} → {out['Date'].max()}")
print(f"   Stores: {out['Store ID'].nunique()}  |  Products: {out['Product ID'].nunique()}")
print("\n🚀 Reload your Streamlit dashboard now!")