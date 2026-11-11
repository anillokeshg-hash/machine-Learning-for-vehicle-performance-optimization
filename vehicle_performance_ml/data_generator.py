"""Generates synthetic vehicle trip data (replace with real telematics/OBD data)."""
import numpy as np
import pandas as pd

FEATURES = ["engine_cc", "vehicle_weight", "load_kg", "avg_speed",
            "tire_pressure", "ambient_temp", "aggressive_driving", "ac_on"]
TARGET = "fuel_efficiency"   # km per litre


def generate(n=3000, seed=42, path="vehicle_data.csv"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "engine_cc": rng.integers(1000, 3001, n),
        "vehicle_weight": rng.integers(900, 2201, n),        # kg
        "load_kg": rng.integers(0, 401, n),                  # passengers + cargo
        "avg_speed": rng.uniform(30, 120, n),                # km/h
        "tire_pressure": rng.uniform(26, 40, n),             # psi
        "ambient_temp": rng.uniform(5, 40, n),               # deg C
        "aggressive_driving": rng.uniform(0, 1, n),          # 0 = smooth, 1 = harsh
        "ac_on": rng.integers(0, 2, n),
    })
    # Hidden "physics": efficiency drops with engine size, weight, very low/high speed,
    # under/over-inflated tires, harsh driving, A/C and extreme temperatures.
    eff = (24
           - 0.0035 * df["engine_cc"]
           - 0.004 * (df["vehicle_weight"] + df["load_kg"])
           - 0.0016 * (df["avg_speed"] - 65) ** 2
           - 0.25 * (df["tire_pressure"] - 33).abs()
           - 3.0 * df["aggressive_driving"]
           - 0.8 * df["ac_on"]
           - 0.03 * (df["ambient_temp"] - 22).abs()
           + rng.normal(0, 0.5, n))
    df[TARGET] = eff.clip(lower=3).round(2)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate()
    print("Saved vehicle_data.csv")
