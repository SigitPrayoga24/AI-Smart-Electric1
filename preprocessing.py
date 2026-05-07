import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('household_power_consumption.txt', sep=';', na_values='?')

# Buat kolom datetime
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Drop baris yang ada NaN
df.dropna(inplace=True)

# Konversi kolom ke numerik
cols = ['Global_active_power','Global_reactive_power',
        'Voltage','Global_intensity',
        'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[cols] = df[cols].apply(pd.to_numeric)

# Agregasi per hari
df['date'] = df['datetime'].dt.date
daily = df.groupby('date').agg(
    total_power   = ('Global_active_power', 'sum'),
    avg_voltage   = ('Voltage', 'mean'),
    avg_intensity = ('Global_intensity', 'mean')
).reset_index()

# Tambah fitur waktu
daily['date']        = pd.to_datetime(daily['date'])
daily['day_of_week'] = daily['date'].dt.dayofweek
daily['month']       = daily['date'].dt.month
daily['day_of_year'] = daily['date'].dt.dayofyear

daily.to_csv('daily_power.csv', index=False)
print("Selesai! Shape:", daily.shape)