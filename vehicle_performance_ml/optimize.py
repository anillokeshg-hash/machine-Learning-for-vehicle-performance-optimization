"""Optimization: search for the driving/tire settings that maximize predicted efficiency."""
import itertools

import numpy as np
import pandas as pd

from data_generator import FEATURES

SPEEDS = np.arange(50, 101, 5)            # km/h
PRESSURES = np.arange(28, 39, 1)          # psi
AGGRESSION = np.round(np.arange(0, 0.51, 0.1), 1)   # keep within realistic smooth driving

TYPICAL_HABITS = {"avg_speed": 90, "tire_pressure": 30, "aggressive_driving": 0.6}


def predict_one(model, vehicle):
    return float(model.predict(pd.DataFrame([vehicle])[FEATURES])[0])


def optimize_settings(model, vehicle):
    """Grid search over controllable settings; vehicle specs and A/C stay fixed."""
    candidates = [{**vehicle, "avg_speed": s, "tire_pressure": p, "aggressive_driving": a}
                  for s, p, a in itertools.product(SPEEDS, PRESSURES, AGGRESSION)]
    preds = model.predict(pd.DataFrame(candidates)[FEATURES])
    best = int(np.argmax(preds))

    baseline_vehicle = {**vehicle, **TYPICAL_HABITS}
    base_eff = predict_one(model, baseline_vehicle)
    best_eff = float(preds[best])
    return {
        "recommended": {k: float(candidates[best][k])
                        for k in ("avg_speed", "tire_pressure", "aggressive_driving")},
        "baseline_km_per_l": round(base_eff, 2),
        "optimized_km_per_l": round(best_eff, 2),
        "improvement_pct": round(100 * (best_eff - base_eff) / base_eff, 1),
        "litres_per_100km_saved": round(100 / base_eff - 100 / best_eff, 2),
    }
