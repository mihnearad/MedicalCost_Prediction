import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load and preprocess data
df = pd.read_csv('medical_costs.csv')

# Identify numeric and categorical columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop('Medical Cost')
categorical_columns = df.select_dtypes(include=['object']).columns

# Encode categorical features and print mappings
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f'Mappings for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}')

# Feature scaling only on numerical columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split dataset
X = df.drop('Medical Cost', axis=1)
y = df['Medical Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model with hyperparameter tuning
model = XGBRegressor(tree_method='hist', gpu_id=0)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model
joblib.dump(best_model, 'medical_cost_model.pkl')
