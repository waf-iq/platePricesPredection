import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
import json

# Load dataset
df = pd.read_csv("../data/enhanced_plates_data.csv")

# Handle categorical variables
categorical_cols = ["emirate", "character", "pattern"]  # Add more if needed
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for inverse transform later

# Define features and target
X = df.drop(columns=["price", "duration", "timestamp", "number", "random"])  # Remove unwanted columns
y = df["price"]  # Target variable (price)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data into DMatrix (XGBoost's optimized format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the objective function for Optuna optimization
def objective(trial):
    param = {
        'objective': 'reg:squarederror',  # Loss function for regression
        'eval_metric': 'mae',  # Use R-squared for evaluation
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # Search for optimal learning rate
        'max_depth': trial.suggest_int('max_depth', 10, 20),  # Search for optimal max depth
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),  # Search for optimal min child weight
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e-3),  # Search for optimal gamma
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),  # Search for optimal subsample ratio
        'max_bin': trial.suggest_int('max_bin', 10, 20),  # Search for optimal max bins
        'tree_method': 'hist',  # Use histogram-based tree building like H2O
    }

    num_boost_round = trial.suggest_int('num_boost_round', 500, 50000, step=250)  # Boosting rounds to try out

    # Train the model with the current set of hyperparameters
    model = xgb.XGBRegressor(**param)

    # Using eval_set and early_stopping_rounds to prevent overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Evaluation on test set
        verbose=False                 # Suppress verbose output
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Return the R-squared (since we're trying to maximize it)
    return r2_score(y_test, y_pred)

# Create an Optuna study to optimize the hyperparameters for 500 trials
study = optuna.create_study(direction='minimize')  # Maximize R-squared score
study.optimize(objective, n_trials=100000)  # Run 500 trials for optimization

# Output the best hyperparameters found by Optuna
print("Best hyperparameters:", study.best_params)

# Now use the best hyperparameters found by Optuna to train the final model
best_params = study.best_params

# Train the final model with the optimized parameters
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Make final predictions on test data
y_pred_final = final_model.predict(X_test)

# Calculate evaluation metrics
mae_final = mean_absolute_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

# Calculate Squared Correlation (R² of Pearson correlation)
correlation_matrix_final = np.corrcoef(y_test, y_pred_final)
squared_correlation_final = correlation_matrix_final[0, 1] ** 2

# Calculate Relative Error (Lenient)
relative_error_lenient_final = np.mean(np.abs(y_test - y_pred_final) / np.maximum(y_test, 1))  # Avoid division by zero

# Print evaluation metrics for the final model
print(f"Mean Absolute Error (MAE) after tuning: {mae_final:.2f}")
print(f"R-squared (R²) after tuning: {r2_final:.4f}")
print(f"Squared Correlation after tuning: {squared_correlation_final:.4f}")
print(f"Relative Error (Lenient) after tuning: {relative_error_lenient_final:.4f}")

# Save actual and predicted prices to a CSV file
results_df_final = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred_final})
results_df_final.to_csv("../results/predicted_prices_xgb_tuned.csv", index=False)
print("Tuned predictions saved to '../results/predicted_prices_xgb_tuned.csv'")

# Save the best hyperparameters as a JSON file
with open("../results/best_hyperparameters.json", "w") as f:
    json.dump(best_params, f)
print("Best hyperparameters saved to '../results/best_hyperparameters.json'")

import joblib

# Save the trained model to a file
joblib.dump(final_model, 'xgb_model.pkl')
print("Model saved as 'xgb_model.pkl'")