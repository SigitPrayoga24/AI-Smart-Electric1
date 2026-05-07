from flask import Flask, render_template
import pandas as pd, json, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('predictions.csv')
    
    # Hitung metrik evaluasi
    mae  = round(mean_absolute_error(df['total_power'], df['predicted']), 2)
    rmse = round(np.sqrt(mean_squared_error(df['total_power'], df['predicted'])), 2)
    r2   = round(r2_score(df['total_power'], df['predicted']), 4)
    avg  = round(df['total_power'].mean(), 1)
    
    # Ambil 60 hari terakhir untuk grafik
    df60      = df.tail(60)
    dates     = df60['date'].tolist()
    actual    = [float(x) for x in df60['total_power'].tolist()]
    predicted = [float(x) for x in df60['predicted'].tolist()]
    
    # Decision support — bandingkan rata-rata 7 hari terakhir vs keseluruhan
    avg_last7  = round(df.tail(7)['total_power'].mean(), 1)
    if avg_last7 > avg * 1.1:
        status = 'tinggi'
    elif avg_last7 < avg * 0.9:
        status = 'rendah'
    else:
        status = 'normal'

    return render_template('index.html',
                           dates=json.dumps(dates),
                           actual=json.dumps(actual),
                           predicted=json.dumps(predicted),
                           mae=mae, rmse=rmse, r2=r2,
                           avg=avg, avg_last7=avg_last7,
                           status=status,
                           total=len(df))

if __name__ == '__main__':
    app.run(debug=True)