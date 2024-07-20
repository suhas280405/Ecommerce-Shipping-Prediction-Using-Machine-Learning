from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('/Users/shanmukhanandudu/Downloads/train (3).csv')

# Preprocess data
le = LabelEncoder()

categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Reached.on.Time_Y.N'], axis=1)
y = df['Reached.on.Time_Y.N']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting classifier
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

@app.route('/', methods=['GET'])
def index():
    return '''
        <h1>Predict Reached.on.Time_Y.N</h1>
        <form id="predict-form" method="POST" action="/predict">
            <label for="ID">ID:</label><br>
            <input type="number" id="ID" name="ID"><br>
            <label for="Warehouse_block">Warehouse Block:</label><br>
            <input type="text" id="Warehouse_block" name="Warehouse_block"><br>
            <label for="Mode_of_Shipment">Mode of Shipment:</label><br>
            <input type="text" id="Mode_of_Shipment" name="Mode_of_Shipment"><br>
            <label for="Customer_care_calls">Customer Care Calls:</label><br>
            <input type="number" id="Customer_care_calls" name="Customer_care_calls"><br>
            <label for="Customer_rating">Customer Rating:</label><br>
            <input type="number" id="Customer_rating" name="Customer_rating"><br>
            <label for="Cost_of_the_Product">Cost of the Product:</label><br>
            <input type="number" id="Cost_of_the_Product" name="Cost_of_the_Product"><br>
            <label for="Prior_purchases">Prior Purchases:</label><br>
            <input type="number" id="Prior_purchases" name="Prior_purchases"><br>
            <label for="Product_importance">Product Importance:</label><br>
            <input type="text" id="Product_importance" name="Product_importance"><br>
            <label for="Gender">Gender:</label><br>
            <input type="text" id="Gender" name="Gender"><br>
            <label for="Discount_offered">Discount offered:</label><br>
            <input type="number" id="Discount_offered" name="Discount_offered"><br>
            <label for="Weight_in_gms">Weight in gms:</label><br>
            <input type="number" id="Weight_in_gms" name="Weight_in_gms"><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Convert input data to a pandas DataFrame
    input_data = pd.DataFrame([data])
    # Encode categorical columns using the same label encoder, handle unseen labels
    for col in categorical_cols:
        try:
            input_data[col] = le.transform(input_data[col])
        except ValueError:
            input_data[col] = -1  # Assign a special value for unseen labels
    # Make prediction using the trained model
    prediction = clf.predict(input_data)
    result = 'On Time' if prediction[0] == 1 else 'Not On Time'
    return render_template_string('''
        <h1>Prediction Result</h1>
        <p>The shipment is predicted to be: <strong>{{ result }}</strong></p>
        <a href="/">Go back to the form</a>
    ''', result=result)

if __name__ == '__main__':
    app.run(debug=True)