# FLIGHT-FARE-PREDICTION



✈️ SkyPredict: Intelligent Flight Fare Estimation

SkyPredict is a machine learning-powered web application that predicts flight ticket prices based on user inputs like airline, route, journey date, duration, and stops. The app is built using Streamlit and leverages Ensemble Learning (Random Forest Regressor) for accurate predictions.

🚀 Features
🎯 Predicts flight fares instantly
🧠 Uses Random Forest (Ensemble Learning)
🎨 Modern UI with premium styling
📊 Considers multiple factors:
Airline
Source & Destination
Journey Date
Duration
Number of Stops
Departure Time
🛠️ Tech Stack
Frontend/UI: Streamlit
Backend: Python
ML Model: Random Forest Regressor
Libraries Used:
pandas
numpy
scikit-learn
joblib
📂 Project Structure
├── app.py                          # Streamlit web application :contentReference[oaicite:0]{index=0}
├── Ensemble_Learning_Techniques.ipynb  # Model training notebook
├── model.pkl                       # Trained ML model
├── scaler.pkl                      # Feature scaler
├── model_features.pkl              # Feature names for prediction
└── README.md                       # Project documentation
⚙️ How It Works
User inputs flight details in the web app.
Input data is:
Preprocessed
One-hot encoded
Scaled using a pre-trained scaler
The trained Random Forest model predicts the fare.
Result is displayed instantly.
