from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.svm import SVR
import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.svm import SVR


app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3307",
    database='fuel'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    if request.method == "POST":
        algorithm_type = request.form['algorithm']
        accuracy = None
        
        if algorithm_type == 'RandomForestRegressor':
            mse = 1.023512  
            mae = 0.812123
        elif algorithm_type == 'RidgeRegression':
            mse = 1.248039
            mae = 0.878757  
        elif algorithm_type == 'AdaBoostRegressor':
            mse = 1.071188  
            mae = 0.829917 
        elif algorithm_type == 'XGBRegressor':
            mse =  1.14826  
            mae =  0.861614  
        elif algorithm_type == 'SVR':
            mse = 1.057808  
            mae = 0.817979  
        else:
            mse = 0  
            mae = 0

        accuracy = {
            'Mean Squared Error': round(mse, 2),
            'Mean Absolute Error': round(mae, 2),
        }
        
        return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)
    
    return render_template('algorithm.html')






# Load dataset
df = pd.read_csv('dataset.csv')

# Encode categorical features
categorical_features = ["traffic_conditions", "weather_conditions", "air_conditioning", "fuel_type"]
label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# Define numerical features
numerical_features = [
    "vehicle_speed_kmh", "engine_speed_rpm", "throttle_position_pct", "engine_load_pct",
    "fuel_rate_lph", "fuel_consumption_l_per_100km", "brake_usage_pct", "acceleration_mps2",
    "deceleration_mps2", "gear_position", "torque_nm", "mass_air_flow_gps", "distance_travelled_km",
    "road_gradient_pct", "vehicle_weight_kg", "tire_pressure_psi"
]

# Ensure driver_profile is categorical
df["driver_profile"] = pd.qcut(df["fuel_consumed_liters"], q=5, labels=[1, 2, 3, 4, 5])

# Apply scaling to numerical columns
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features and targets
X = df[numerical_features + categorical_features]
y_reg = df["fuel_consumed_liters"]

# Split dataset
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_reg_train)

#ridge_model = Ridge(alpha=1.0)
#ridge_model.fit(X_train, y_reg_train)

# Define driver profile mapping dynamically
fuel_profile = df.groupby("driver_profile")["fuel_consumed_liters"].agg(["min", "max"]).reset_index()
fuel_profile.columns = ["Driver Profile", "Min Fuel Consumed (L)", "Max Fuel Consumed (L)"]

# Function to classify driver profile based on fuel consumption
def classify_driver_profile(fuel_consumed):
    profile = fuel_profile[
        (fuel_profile["Min Fuel Consumed (L)"] <= fuel_consumed) & 
        (fuel_profile["Max Fuel Consumed (L)"] >= fuel_consumed)
    ]["Driver Profile"].values
    return int(profile[0]) if len(profile) > 0 else 5  # Default to most aggressive if out of range

# Function to get driving suggestions
def get_suggestion(profile):
    suggestions = {
        1: "Eco:Amazing! Your driving is extremely fuel-efficient. Keep it up!",
        2: "Normal:Good job! Try to maintain steady speeds and smooth acceleration.",
        3: "Calm:Your driving is balanced. Consider optimizing fuel efficiency by reducing idling.",
        4: "Sporty:Your driving consumes more fuel. Avoid rapid acceleration and hard braking.",
        5: "Aggressive:Warning! Your driving is highly fuel-inefficient. Consider smoother driving habits."
    }
    return suggestions.get(profile, "No suggestion available.")

    

@app.route('/predict_fuel_driver', methods=['POST', 'GET'])
def predict_fuel_driver():
    if request.method == 'POST':
        # Retrieve input values dynamically from the form
        input_values = []
        feature_names = ["vehicle_speed_kmh", "engine_speed_rpm", "throttle_position_pct", "engine_load_pct",
                         "fuel_rate_lph", "brake_usage_pct", "acceleration_mps2", "deceleration_mps2",
                         "gear_position", "torque_nm", "distance_travelled_km", "road_gradient_pct",
                         "vehicle_weight_kg", "tire_pressure_psi", "traffic_conditions", "weather_conditions",
                         "air_conditioning", "fuel_type"]

        feature_names = numerical_features + categorical_features

        for feature in feature_names:
            input_values.append(float(request.form.get(feature, 0)))  # Convert input to float, default to 0 if missing

        # Convert input list to numpy array
        input_data = np.array([input_values])

        # Scale ONLY numerical features before prediction
        input_data[:, :len(numerical_features)] = scaler.transform(input_data[:, :len(numerical_features)])

        # Predict fuel consumption using the trained Random Forest model
        fuel_consumed = rf_model.predict(input_data)

        # Format the predicted fuel consumption value
        fuel_consumed_formatted = round(fuel_consumed[0], 2)

        # Classify the driver profile based on predicted fuel consumption
        driver_profile = classify_driver_profile(fuel_consumed_formatted)

        # Get a suggestion based on the classified profile
        suggestion = get_suggestion(driver_profile)

        # Render the prediction template with results
        return render_template(
            'prediction.html',
            prediction=fuel_consumed_formatted,
            driver_profile=driver_profile,
            suggestion=suggestion
        )

    return render_template('prediction.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)