"""Machine Learning for Vehicle Performance Optimization - run this file."""
import json
import os

import pandas as pd

from data_generator import generate
from optimize import TYPICAL_HABITS, optimize_settings
from train_model import plot_report, train_and_evaluate

SAMPLE_VEHICLES = {
    "Hatchback": {"engine_cc": 1200, "vehicle_weight": 950, "load_kg": 100,
                  "ambient_temp": 25, "ac_on": 1},
    "Sedan": {"engine_cc": 1800, "vehicle_weight": 1300, "load_kg": 200,
              "ambient_temp": 32, "ac_on": 1},
    "SUV": {"engine_cc": 2800, "vehicle_weight": 2000, "load_kg": 300,
            "ambient_temp": 18, "ac_on": 0},
}


def main():
    df = pd.read_csv("vehicle_data.csv") if os.path.exists("vehicle_data.csv") else generate()
    print(f"Dataset: {len(df)} trips\n")

    best_name, model, results, (X_te, y_te) = train_and_evaluate(df)
    print("=== MODEL COMPARISON (higher R2 is better) ===")
    print(results.round(3).to_string(index=False))
    print(f"\nBest model: {best_name} (saved to best_model.joblib)")

    imp = plot_report(best_name, model, X_te, y_te)
    print("\n=== TOP FACTORS AFFECTING FUEL EFFICIENCY ===")
    print(imp.head(5).round(3).to_string())

    print(f"\n=== OPTIMIZATION (typical habits: {TYPICAL_HABITS}) ===")
    report = {}
    for name, spec in SAMPLE_VEHICLES.items():
        r = optimize_settings(model, spec)
        report[name] = r
        rec = r["recommended"]
        print(f"\n{name}:")
        print(f"  Recommended -> speed {rec['avg_speed']:.0f} km/h, "
              f"tire pressure {rec['tire_pressure']:.0f} psi, "
              f"driving aggression {rec['aggressive_driving']}")
        print(f"  Efficiency  -> {r['baseline_km_per_l']} km/l to {r['optimized_km_per_l']} km/l "
              f"(+{r['improvement_pct']}%), saves {r['litres_per_100km_saved']} L per 100 km")

    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved: model_report.png, optimization_report.json, best_model.joblib")


if __name__ == "__main__":
    main()
