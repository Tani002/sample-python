from flask import Flask, render_template, request, session, redirect, json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pyrebase

# Load the model from the file
with open('ARIMA_VP.pkl', 'rb') as f:
    model_vp = pickle.load(f)

with open('ARIMA_AH.pkl', 'rb') as f:
    model_ah = pickle.load(f)

with open('ARIMA_FP.pkl', 'rb') as f:
    model_fp = pickle.load(f)

app = Flask(__name__,static_folder='static')

config = {
    'apiKey': "AIzaSyBixtA4v5mvxKvaTU61iq9Fr2Ln2OWlf3o",
    'authDomain': "tomatocare-78e23.firebaseapp.com",
    'projectId': "tomatocare-78e23",
    'storageBucket': "tomatocare-78e23.appspot.com",
    'messagingSenderId': "437959910172",
    'appId': "1:437959910172:web:dab8c80225929289dd90d9",
    'databaseURL' : "",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

app.secret_key='secret'

@app.route('/')
def home_route():
    return render_template('index.html')

# Define the route for the forecasting page
@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    if request.method == 'POST':
        # Retrieve user input from the form
        num_years = int(request.form['num_years'])

        # Step 1: Load the data
        data = pd.read_csv('AreaHarvested.csv')

        # Convert 'Year' column to string type
        data['Year'] = data['Year'].astype(str)
        data['YearQuarter'] = data['Year'] + '-' + data['TimePeriod']

        data['AreaHarvested_log'] = np.log(data['AreaHarvested'])
        train_data = data['AreaHarvested_log'].iloc[:int(len(data) * 0.7)]
        train_data_diff = train_data.diff().dropna()

        # Step 3: Fit the ARIMA model
        model = ARIMA(train_data, order=(4, 1, 0))
        model_fit = model.fit()

        # Step 4: Make time series predictions
        test_data = data['AreaHarvested_log'].iloc[int(len(data) * 0.7):]
        forecast = model_fit.forecast(steps=len(test_data))

        # Step 2: Fit the ARIMA model
        model = ARIMA(data['AreaHarvested_log'], order=(4, 1, 0))
        model_fit = model.fit()

        combined_data = pd.concat([data[['YearQuarter', 'AreaHarvested_log']], pd.Series(forecast)], axis=1)
        combined_data.columns = ['Year', 'Actual', 'Forecast']

        # Fit the ARIMA model using the actual series
        train_data = data['AreaHarvested'].iloc[:int(len(data) * 0.7)]
        model = ARIMA(train_data, order=(4, 1, 0))
        model_fit = model.fit()

        # Step 3: Make time series predictions
        last_year = int(data['Year'].iloc[-1])
        last_year = last_year + 1
        future_years = pd.date_range(start=f'{last_year}-01-01', periods=num_years * 4, freq='Q')
        forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

        # Step 4: Create a DataFrame with the predicted data
        prediction_df = pd.DataFrame({
            'Year': future_years.year,
            'TimePeriod': future_years.quarter,
            'Actual': data['AreaHarvested'].values[-num_years * 4:],
            'Forecast': forecast
        })

        # Render the forecasting_results.html template with the predicted data
        return render_template('forecasting_results.html', prediction_df=prediction_df.to_dict(orient='records'))

     # Render the forecasting.html template for user input
    return render_template('forecasting.html')

@app.route('/',methods=['POST','GET'])
def login():
    # if "user" in session:
    #     print("Redirecting to /index")
    #     return redirect('/index')
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print("Email:", email)
        print("Password:", password)
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
            print("User session set:", session['user'])
            return redirect('/index')
        except:
            error = 'Failed to login. Invalid email or password.' 
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/login')

@app.route('/login')
def user_login():
    return render_template('login.html')

@app.route('/profile')
def user_profile():
    if "user" not in session:
        return redirect('login')
    user_email = session['user']
    user_info = {
        'name': 'John Doe',
        'age': 30,
        'email': user_email,
        # Add more fields as needed
    }

    return render_template('profile.html', user_info=user_info)

@app.route('/index')
def index():
    print(session)
    if "user" not in session:
        return redirect('/login')
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            session['user'] = user
            return redirect('/login')
        except Exception as e:
            error = str(e)
            return render_template('signup.html', error=error)
    return render_template('signup.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)


# @app.route('/')
# def default_route():
#     return render_template('login.html')

# @app.route('/')
# def home_route():
#     return render_template('index.html')


# @app.route('/', methods=['POST', 'GET'])
# def login():
#     if "user" in session:
#         return redirect('/index')

#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         try:
#             user = auth.sign_in_with_email_and_password(email, password)
#             session['user'] = email
#             return redirect('/index')
#         except:
#             return 'Failed to login'
#     return render_template('login.html')


# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if "user" in session:
#         return redirect('/login')

#     if request.method == 'POST':
#         try:
#             email = request.form.get('email')
#             password = request.form.get('password')
#             user = auth.create_user_with_email_and_password(email, password)
#             auth.send_email_verification(user['idToken'])
#             session['user'] = email
#             return redirect('/login')
#         except Exception as e:
#             error = str(e)
#             return render_template('signup.html', error=error)
#     return render_template('signup.html')


# @app.route('/index')
# def index():
#     if "user" not in session:
#         return redirect('/login')
#     return render_template('index.html')

# @app.route('/logout')
# def logout():
#     session.pop('user')
#     return redirect('/login')