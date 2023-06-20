from flask import Flask, render_template, request, session, redirect
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore, auth, exceptions

app = Flask(__name__)
app.secret_key = "secret"

# Load the ARIMA models
with open("ARIMA/ARIMA_VP.pkl", "rb") as f:
    model_vp = pickle.load(f)

with open("ARIMA/ARIMA_AH.pkl", "rb") as f:
    model_ah = pickle.load(f)

with open("ARIMA/ARIMA_FP.pkl", "rb") as f:
    model_fp = pickle.load(f)

# Initialize Firebase app
config = {
    "apiKey": "AIzaSyBixtA4v5mvxKvaTU61iq9Fr2Ln2OWlf3o",
    "authDomain": "tomatocare-78e23.firebaseapp.com",
    "projectId": "tomatocare-78e23",
    "storageBucket": "tomatocare-78e23.appspot.com",
    "messagingSenderId": "437959910172",
    "appId": "1:437959910172:web:dab8c80225929289dd90d9",
    "databaseURL": "",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Initialize the app with the service account credentials
cred = credentials.Certificate(
    "tomatocare-78e23-firebase-adminsdk-by348-b01efacff5.json"
)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Routes


# Define the route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Define the route for the forecasting page
@app.route("/forecasting", methods=["GET", "POST"])
def forecasting():
    if request.method == "POST":
        # Retrieve user input from the form
        num_years = int(request.form["num_years"])
        selected_option = request.form["selected_option"]
        number = request.form.get("num_years")

        if selected_option == "AreaHarvested":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/AreaHarvested.csv")

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["AreaHarvested_log"] = np.log(data["AreaHarvested"])
            train_data = data["AreaHarvested_log"].iloc[: int(len(data) * 0.7)]
            train_data_diff = train_data.diff().dropna()

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["AreaHarvested_log"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA modelg
            model = ARIMA(data["AreaHarvested_log"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "AreaHarvested_log"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["AreaHarvested"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["AreaHarvested"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_AreaResults.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

        elif selected_option == "VolumeProduction":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/VolumeProduction.csv")

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["VolumeProduction_log"] = np.log(data["VolumeProduction"])
            train_data = data["VolumeProduction_log"].iloc[: int(len(data) * 0.7)]
            train_data_diff = train_data.diff().dropna()

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["VolumeProduction_log"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA model
            model = ARIMA(data["VolumeProduction_log"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "VolumeProduction_log"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["VolumeProduction"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["VolumeProduction"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_volumeResult.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

        elif selected_option == "FarmgatePrices":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/FarmgatePrices.csv")

            # To avoid using scientific notation
            pd.set_option("display.float_format", lambda x: "%d" % x)

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["FarmgatePrices"] = np.log(data["FarmgatePrices"])
            train_data = data["FarmgatePrices"].iloc[: int(len(data) * 0.7)]
            train_data_diff = train_data.diff().dropna()

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["FarmgatePrices"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA model
            model = ARIMA(data["FarmgatePrices"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "FarmgatePrices"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["FarmgatePrices"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["FarmgatePrices"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_pricesResult.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

        elif selected_option == "VolumeDemand":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/VolumeDemand.csv")

            # To avoid using scientific notation
            pd.set_option("display.float_format", lambda x: "%d" % x)

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["VolumeDemand"] = np.log(data["VolumeDemand"])
            train_data = data["VolumeDemand"].iloc[: int(len(data) * 0.7)]
            train_data_diff = train_data.diff().dropna()

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["VolumeDemand"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA model
            model = ARIMA(data["VolumeDemand"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "VolumeDemand"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["VolumeDemand"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["VolumeDemand"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_volumeDemandResult.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

    # Render the forecasting.html template for user input
    return render_template("forecasting.html")


# @app.route('/submit', methods=['POST'])
# def submit(prediction_df):
#     #Process the form submission and generate the results
#      max_area = request.form.get['max_area']
#      forecast_years = request.form.get('forecast_years')

#     # Pass the results to the template for rendering
#      forecast_results = {
#      'max_area': max_area,
#      'forecast_years': forecast_years,
#      'prediction_df': prediction_df.to_dict('records')
#     }
#      return render_template('index.html', forecast_results=forecast_results)


# User login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Handle login form submission
        email = request.form["email"]
        password = request.form["password"]
        try:
            # Sign in the user with email and password
            user = auth.sign_in_with_email_and_password(email, password)
            # Store user session data
            session["user"] = user
            return redirect("/")
        except:
            error = "Invalid email or password"
            return render_template("login.html", error=error)
    else:
        # Display login form
        return render_template("login.html")


@app.route("/logout")
def logout():
    # Clear user session data
    session.pop("user", None)
    return redirect("/login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # Handle signup form submission
        email = request.form["email"]
        password = request.form["password"]
        username = request.form["username"]
        # Create a new user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)
        # Store user details in Firestore database
        db.collection("users").document(user["localId"]).set(
            {"email": email, "name": username, "emailVerified": False}
        )
        # Send email verification link to the user
        auth.send_email_verification(user["idToken"])
        return redirect("/login")
    else:
        return render_template("signup.html")


@app.route("/profile")
def user_profile():
    if "user" in session:
        user = session["user"]
        # Fetch user profile data from Firestore based on user ID
        profile = db.collection("users").document(user["localId"]).get().to_dict()
        return render_template("profile.html", profile=profile)
    else:
        return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)


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
