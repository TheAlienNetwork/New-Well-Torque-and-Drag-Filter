from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def create_plot(df, bounds_factor):
    plots = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and col != 'MD(ft)':
            df_clean = df[['MD(ft)', col]].dropna()
            df_clean = df_clean[df_clean[col] > 1]
            if df_clean.empty:
                continue

            X = df_clean.index.values.reshape(-1, 1)
            y = df_clean[col].values
            
            if len(y) < 2:
                continue

            poly = PolynomialFeatures(degree=3)
            model = make_pipeline(poly, Ridge())
            model.fit(X, y)
            trend_line = model.predict(X)
            
            residuals = y - trend_line
            std_residuals = np.std(residuals)
            upper_bound = trend_line + bounds_factor * std_residuals
            lower_bound = trend_line - bounds_factor * std_residuals

            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_filtered = df_clean[mask]

            fig_unfiltered = px.scatter(df_clean, x=df_clean.index, y=col, title=f'{col} (Unfiltered)', 
                             template='plotly_dark')
            fig_unfiltered.add_scatter(x=df_clean.index, y=trend_line, mode='lines', name='Trend Line')
            fig_unfiltered.add_scatter(x=df_clean.index, y=upper_bound, mode='lines', name='Upper Bound', line=dict(dash='dash'))
            fig_unfiltered.add_scatter(x=df_clean.index, y=lower_bound, mode='lines', name='Lower Bound', line=dict(dash='dash'))

            plot_html_unfiltered = pio.to_html(fig_unfiltered, full_html=False)
            plot_filename_unfiltered = f'{col.replace(" ", "_")}_unfiltered.html'
            plot_path_unfiltered = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename_unfiltered)
            with open(plot_path_unfiltered, 'w', encoding='utf-8') as file:
                file.write(plot_html_unfiltered)

            fig_filtered = px.scatter(df_filtered, x=df_filtered.index, y=col, title=f'{col} (Filtered)', 
                             template='plotly_dark')
            fig_filtered.add_scatter(x=df_filtered.index, y=trend_line[mask], mode='lines', name='Trend Line')
            fig_filtered.add_scatter(x=df_filtered.index, y=upper_bound[mask], mode='lines', name='Upper Bound', line=dict(dash='dash'))
            fig_filtered.add_scatter(x=df_filtered.index, y=lower_bound[mask], mode='lines', name='Lower Bound', line=dict(dash='dash'))

            plot_html_filtered = pio.to_html(fig_filtered, full_html=False)
            plot_filename_filtered = f'{col.replace(" ", "_")}_filtered.html'
            plot_path_filtered = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename_filtered)
            with open(plot_path_filtered, 'w', encoding='utf-8') as file:
                file.write(plot_html_filtered)

            plots[col] = {
                'filtered': plot_filename_filtered,
                'unfiltered': plot_filename_unfiltered
            }
    return plots

def compute_trend_accuracy(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2 * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    global raw_file_path
    file_info = None
    plots = {}
    bounds_factor = 1.0

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file:
                file_size = len(file.read())
                file.seek(0)
                file_name = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                file.save(filepath)
                raw_file_path = filepath  # Save the raw file path globally

                df = pd.read_csv(filepath, encoding='utf-8')

                filtered_df = df.copy()
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64'] and col != 'MD(ft)':
                        df_clean = df[['MD(ft)', col]].dropna()
                        df_clean = df_clean[df_clean[col] > 1]

                        X = df_clean.index.values.reshape(-1, 1)
                        y = df_clean[col].values

                        if len(y) < 2:
                            continue
                        
                        poly = PolynomialFeatures(degree=3)
                        model = make_pipeline(poly, Ridge())
                        model.fit(X, y)
                        trend_line = model.predict(X)
                        residuals = y - trend_line
                        std_residuals = np.std(residuals)
                        upper_bound = trend_line + bounds_factor * std_residuals
                        lower_bound = trend_line - bounds_factor * std_residuals

                        bounds_upper = np.full(df.shape[0], np.nan)
                        bounds_lower = np.full(df.shape[0], np.nan)
                        
                        bounds_upper[df_clean.index] = upper_bound
                        bounds_lower[df_clean.index] = lower_bound

                        mask = (df[col] >= bounds_lower) & (df[col] <= bounds_upper)
                        filtered_df[col] = np.where(mask, df[col], np.nan)

                filtered_df = filtered_df.dropna(how='all', subset=[col for col in filtered_df.columns if col != 'MD(ft)'])
                
                filtered_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.csv')
                filtered_df.to_csv(filtered_filepath, index=False, encoding='utf-8')

                plots = create_plot(filtered_df, bounds_factor)

                total_points = len(df)
                filtered_points = len(filtered_df.dropna(how='all', subset=[col for col in filtered_df.columns if col != 'MD(ft)']))
                
                filtered_percentage = (filtered_points / total_points) * 100 if total_points > 0 else 0

                file_info = {
                    'name': file_name,
                    'size': file_size,
                    'filtered_data': 'filtered_data.csv'
                }
                return render_template('index.html', plots=plots, tables=[filtered_df.to_html(classes='table table-striped', index=False)], file_info=file_info, bounds_factor=bounds_factor, filtered_percentage=filtered_percentage)
    
    return render_template('index.html', plots=plots, file_info=file_info, bounds_factor=bounds_factor)

@app.route('/update_plots', methods=['POST'])
def update_plots():
    bounds_factor = float(request.form['bounds_factor'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.csv')
    df = pd.read_csv(filepath, encoding='utf-8')
    plots = create_plot(df, bounds_factor)
    return jsonify(plots)

@app.route('/plot/<plot_name>')
def view_plot(plot_name):
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_name)
    if os.path.exists(plot_path):
        with open(plot_path, 'r', encoding='utf-8') as file:
            plot_html = file.read()
        return render_template('plot.html', plot=plot_html)
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/diagnostics')
def diagnostics():
    return render_template('diagnostics.html')

@app.route('/diagnostics_data')
def diagnostics_data():
    global raw_file_path
    if not raw_file_path:
        return jsonify({'filtered_percentage': 0, 'trend_accuracy': 0})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.csv')
    df = pd.read_csv(filepath, encoding='utf-8')

    total_points = len(pd.read_csv(raw_file_path))
    filtered_points = df.dropna(how='all', subset=[col for col in df.columns if col != 'MD(ft)']).shape[0]
    
    filtered_percentage = (filtered_points / total_points) * 100 if total_points > 0 else 0
    trend_accuracy = 0
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and col != 'MD(ft)':
            df_clean = df[['MD(ft)', col]].dropna()
            df_clean = df_clean[df_clean[col] > 1]
            if df_clean.empty:
                continue

            X = df_clean.index.values.reshape(-1, 1)
            y = df_clean[col].values
            if len(y) < 2:
                continue

            poly = PolynomialFeatures(degree=3)
            model = make_pipeline(poly, Ridge())
            model.fit(X, y)
            trend_line = model.predict(X)

            accuracy = compute_trend_accuracy(y, trend_line)
            trend_accuracy = max(trend_accuracy, accuracy)

    diagnostics = {
        'filtered_percentage': filtered_percentage,
        'trend_accuracy': trend_accuracy
    }

    return jsonify(diagnostics)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable provided by Render
    app.run(host='0.0.0.0', port=port, debug=True)
