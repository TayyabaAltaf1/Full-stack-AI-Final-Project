import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Step 1: Data Load ---
df = pd.read_csv(r'C:\Users\Lataisha\Downloads\loan_data.csv')

# --- Step 2: Categorical Data Handling (One-Hot Encoding for 'purpose') ---
final_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)

# --- Step 3: Train/Test Split ---
X = final_df.drop('not.fully.paid', axis=1) # Features
y = final_df['not.fully.paid']             # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# --- Step 4: Model Training (Random Forest) ---
model_ml = RandomForestClassifier(n_estimators=100, random_state=101)
model_ml.fit(X_train, y_train)

# --- Step 5: Evaluation ---
predictions = model_ml.predict(X_test)
print("--- Classification Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# --- Feature Importance (Extra Marks ke liye) ---
print("\n--- Top 5 Important Features ---")
feature_importances = pd.Series(model_ml.feature_importances_, index=X.columns)
print(feature_importances.nlargest(5))