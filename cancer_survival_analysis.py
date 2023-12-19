# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load my Example data for project
# df = pd.read_csv('example_dataset.csv')

# Example data for project
data = {
    'Age': [55, 63, 48], 
    'Gender': ['Female', 'Male', 'Female'],
    'Ethnicity': ['Caucasian', 'African', 'Asian'],
    'Cancer Type': ['Breast', 'Lung', 'Colorectal'],
    'Stage': ['IV', 'IV', 'IV'],
    'Survival Status': ['Alive', 'Alive', 'Deceased']
}
df = pd.DataFrame(data)

# Data Preprocessing
# Encoding categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Splitting the dataset into features and target variable
X = df.drop('Survival Status', axis=1)
y = df['Survival Status']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling for numerical stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Parameters from Grid Search
print(f"Best Parameters: {grid_search.best_params_}")

# Training the model with best parameters
best_rf = grid_search.best_estimator_

# Cross-validation for robustness check
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")

# Evaluating the model on the test set
y_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generating and plotting confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Analyzing feature importances
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Optionally, save the trained model
# import joblib
# joblib.dump(best_rf, 'advanced_cancer_survival_rf_model.pkl')
