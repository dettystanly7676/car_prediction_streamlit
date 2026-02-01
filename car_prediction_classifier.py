import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load the dataset
df = pd.read_csv('car_prediction_data.csv')

# 2. Feature Engineering: Create 'Age' and drop unnecessary columns
df['Age'] = 2024 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# 3. Handle Categorical Data (Fuel, Seller, Transmission)
# This converts text into 0s and 1s
df = pd.get_dummies(df, drop_first=True)

# 4. Split data into Features (X) and Target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# 5. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Save the model AND the column names (This is the secret to avoiding errors!)
model_data = {
    "model": model,
    "columns": X.columns.tolist()
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ model.pkl created successfully with all categories!")
print(f"Features used: {X.columns.tolist()}")