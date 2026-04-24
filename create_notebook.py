import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    cells = []
    
    # 1. Theory Markdown
    markdown_theory = """# Ensemble Learning Techniques & Flight Fare Prediction

## Task Objectives
1. Learn the basic concept of Ensemble Learning
2. Study different types of Ensemble Techniques (Bagging, Boosting, Stacking)
3. Implement the Random Forest algorithm
4. Use a dataset (Flight Fare Prediction)
5. Train and test 10 ML models including Random Forest
6. Evaluate model performance and compare results

---
## 1. Theory of Ensemble Learning
**Ensemble Learning** is a machine learning paradigm where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

### Types of Ensemble Techniques:
- **Bagging (Bootstrap Aggregating)**: Fits multiple independent models in parallel and averages their predictions. **Example:** Random Forest.
- **Boosting**: Fits multiple models sequentially. Each model attempts to correct the errors of its predecessor. **Examples:** Gradient Boosting, AdaBoost, XGBoost.
- **Stacking**: Combines multiple heterogeneous base models using a meta-model that outputs the final prediction based on the base models' predictions.
"""
    cells.append(nbf.v4.new_markdown_cell(markdown_theory))

    # 2. Setup Code
    code_setup = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # 3. Data Loading and Preprocessing Markdown
    markdown_data = "## 2. Load and Preprocess Flight Fare Data\nWe load the generated `flight_fare_dataset.csv` and apply feature engineering (dates, duration, one-hot encoding)."
    cells.append(nbf.v4.new_markdown_cell(markdown_data))

    # 4. Data Loading and Preprocessing Code
    code_data = """# Load Data
df = pd.read_csv('flight_fare_dataset.csv')
print("Dataset Head:")
display(df.head())

# Preprocessing
df.dropna(inplace=True)
df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"], axis=1, inplace=True)

df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis=1, inplace=True)

duration = list(df["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:    
        if "h" in duration[i]: duration[i] = duration[i].strip() + " 0m"   
        else: duration[i] = "0h " + duration[i]
            
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
    
df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins
df.drop(["Duration"], axis=1, inplace=True)

airline = pd.get_dummies(df[["Airline"]], drop_first=True)
source = pd.get_dummies(df[["Source"]], drop_first=True)
destination = pd.get_dummies(df[["Destination"]], drop_first=True)

df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

df_processed = pd.concat([df, airline, source, destination], axis=1)
df_processed.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

X = df_processed.drop('Price', axis=1)
y = df_processed['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
    cells.append(nbf.v4.new_code_cell(code_data))

    # 5. Training 10 Models
    markdown_models = "## 3. Training and Evaluating 10 ML Models\nWe'll train multiple models including our target **Random Forest**, and compare their performance using R2 Score and RMSE."
    cells.append(nbf.v4.new_markdown_cell(markdown_models))

    code_models = """models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "K-Neighbors": KNeighborsRegressor(),
    "Support Vector Regressor": SVR()
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
    
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
display(results_df)
"""
    cells.append(nbf.v4.new_code_cell(code_models))

    # 6. Saving Model
    markdown_save = "## 4. Saving the Random Forest Model\nWe save the trained Random Forest model along with the scaler for our Streamlit frontend application."
    cells.append(nbf.v4.new_markdown_cell(markdown_save))

    code_save = """rf_model = models["Random Forest"]
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
print("Model, scaler, and features saved successfully!")
"""
    cells.append(nbf.v4.new_code_cell(code_save))

    nb['cells'] = cells
    with open('Ensemble_Learning_Techniques.ipynb', 'w') as f:
        nbf.write(nb, f)
        
if __name__ == "__main__":
    create_notebook()
    print("Notebook 'Ensemble_Learning_Techniques.ipynb' created successfully.")
