import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('daily_power.csv')

features = ['avg_voltage','avg_intensity','day_of_week','month','day_of_year']
X = df[features]
y = df['total_power']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    'Linear_Regression': LinearRegression(),
    'Random_Forest':     RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name}:")
    print(f"  MAE  = {mean_absolute_error(y_test, pred):.2f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
    print(f"  R2   = {r2_score(y_test, pred):.4f}")
    joblib.dump(model, f'{name}.pkl')

# Simpan prediksi pakai Random Forest
best = joblib.load('Random_Forest.pkl')
df['predicted'] = best.predict(X).astype(float)

# Verifikasi
print("\nSample predictions.csv:")
print(df[['date','total_power','predicted']].tail(5).to_string())

df[['date','total_power','predicted']].to_csv('predictions.csv', index=False)
print("\nSelesai! predictions.csv tersimpan.")