# Machine Learning for Vehicle Performance Optimization

A simple base project that predicts a vehicle's fuel efficiency (km/l) from its specs and
driving conditions, then recommends the speed, tire pressure and driving style that
maximize efficiency.

## Pipeline
1. **Data** (`data_generator.py`) - creates 3,000 synthetic trips with engine size, weight,
   load, average speed, tire pressure, temperature, driving aggressiveness and A/C use.
2. **Models** (`train_model.py`) - trains Linear Regression, Random Forest and Gradient
   Boosting, compares R2 / MAE / RMSE, saves the best as `best_model.joblib`, and plots
   actual vs predicted plus feature importance (`model_report.png`).
3. **Optimization** (`optimize.py`) - grid-searches the controllable settings (speed, tire
   pressure, driving aggression) using the trained model and reports the gain versus
   typical habits.
4. **Run** (`main.py`) - executes everything and writes `optimization_report.json`.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Use real data
Replace `vehicle_data.csv` with your own file (OBD-II / telematics logs) using the same
columns as `FEATURES` + `fuel_efficiency` in `data_generator.py`. Adjust the feature list
if your data has other signals (RPM, gear, road gradient, etc.).

## Ideas to extend
Predictive maintenance, battery range prediction for EVs, route-level optimization,
hyperparameter tuning (GridSearchCV), a Streamlit app, live OBD-II data via `python-obd`.
