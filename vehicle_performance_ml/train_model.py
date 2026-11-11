"""Train and compare regression models that predict fuel efficiency."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import FEATURES, TARGET


def train_and_evaluate(df):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"model": name,
                     "R2": r2_score(y_te, pred),
                     "MAE": mean_absolute_error(y_te, pred),
                     "RMSE": float(np.sqrt(mean_squared_error(y_te, pred)))})
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]
    best = fitted[best_name]
    joblib.dump(best, "best_model.joblib")
    return best_name, best, results, (X_te, y_te)


def plot_report(best_name, model, X_te, y_te, path="model_report.png"):
    pred = model.predict(X_te)
    imp = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0)
    imp = pd.Series(imp.importances_mean, index=X_te.columns).sort_values()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(y_te, pred, s=8, alpha=0.5)
    lims = [y_te.min(), y_te.max()]
    ax[0].plot(lims, lims, "r--")
    ax[0].set_xlabel("Actual km/l")
    ax[0].set_ylabel("Predicted km/l")
    ax[0].set_title(f"{best_name}: Actual vs Predicted")
    ax[1].barh(imp.index, imp.values, color="teal")
    ax[1].set_title("What affects fuel efficiency most")
    ax[1].set_xlabel("Importance (drop in R2 when shuffled)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    return imp.sort_values(ascending=False)
