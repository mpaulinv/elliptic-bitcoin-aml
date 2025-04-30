### This script will adjust the proposed models to the data 
# ### and evaluate their performance.

# ### The models to be used are:
# ### 1. Random Forest (out of the box)
# ### 2. Random Forest (with hyperparameter tuning)
# ### 3. Random Forest with feature selection (using the top 100 features based on importance)

#before I adjust the models I will load the data 

#load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Define the path to the train_set_clean CSV file
train_set_clean_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\train_set_clean.csv'

# Load the train_set_clean dataset
train_set_clean = pd.read_csv(train_set_clean_path)

# Display the first few rows to verify the data
print("Loaded train_set_clean dataset:")
print(train_set_clean.head())

# Count the number of instances for each class
class_counts = train_set_clean['class'].value_counts()

# Display the counts
print("Number of instances for each class:")
print(class_counts)

# Optionally, display the counts in a more descriptive format
print(f"Class 0 (legitimate): {class_counts.get(0, 0)}")
print(f"Class 1 (ilicit): {class_counts.get(1, 0)}")

# Define the target variable and features
X = train_set_clean.drop(columns=['class', 'transaction_id', 'timestep'])  # Features
y = train_set_clean['class']  # Target variable

# Get the original class counts
class_0_count = class_counts[0]  # Number of samples in class 0
class_1_count = class_counts[1]  # Number of samples in class 1

# Calculate the desired number of samples for class 1
desired_class_1_count = int(class_0_count * (30 / 70))  # 30% of the total

# Define the sampling strategy
desired_ratio = {0: class_0_count, 1: max(class_1_count, desired_class_1_count)}

# Apply RandomOverSampler to achieve the desired ratio
oversampler = RandomOverSampler(sampling_strategy=desired_ratio, random_state=1001)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Combine the resampled features and target into a new DataFrame
train_set_oversampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='class')], axis=1)

# Display the new class distribution
print("New class distribution after oversampling:")
print(train_set_oversampled['class'].value_counts(normalize=True))


# Count the number of instances for each class
class_counts = train_set_oversampled['class'].value_counts()

# Display the counts
print("Number of instances for each class:")
print(class_counts)

# Optionally, display the counts in a more descriptive format
print(f"Class 0 (legitimate): {class_counts.get(0, 0)}")
print(f"Class 1 (ilicit): {class_counts.get(1, 0)}")

### Model: Random Forest.
# ### I will use the Random Forest model from the sklearn library.
# ### In terms of variables, I will use the features plus some of the variables from the combined dataset.
# ### I will exclude num_nodes, num_edges, in_degree_centrality and num_connected_components.
# ### To assess performance I will do a 5-fold cross-validation.
# Hyperparameters for Random Forest tunning 
# n_estimators = 100, max_depth = 10, n_features=50, m random_state = 1001
### Itried   'n_estimators': [40, 50, 60],  # Number of trees
##    'max_depth': [20,25,30],     # Maximum depth of each tree
##    'max_features': [25,40,50]    # Number of features to consider at each split
## Best parameters in terms of f1 score were 20 max depth, 25 mas features and 60 n_estimators.

# Exclude specified features
excluded_features = ['num_nodes', 'num_edges', 'in_degree_centrality', 'num_connected_components']
X_rf = train_set_oversampled.drop(columns=excluded_features + ['class'])  # Features
y_rf = train_set_oversampled['class']  # Target variable

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [20],  # Number of trees
    'max_depth': [25],     # Maximum depth of each tree
    'max_features': [60]    # Number of features to consider at each split
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=1001)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='f1',  # Optimize F1-score for the illicit class
    cv=5,          # 5-fold cross-validation
    verbose=1,     # Display progress
    n_jobs=-1      # Use all available CPU cores
)

# Perform the grid search
grid_search.fit(X_rf, y_rf)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:")
print(best_params)
print(f"Best F1-Score (Cross-Validation): {best_score:.4f}")

# Use cross-validation predictions to evaluate the best model
y_rf_cv_pred = cross_val_predict(grid_search.best_estimator_, X_rf, y_rf, cv=5)

# Generate a classification report based on cross-validation predictions
print("Classification Report for Cross-Validation Predictions:")
print(classification_report(y_rf, y_rf_cv_pred, target_names=["Legitimate (0)", "Illicit (1)"]))

# Display the confusion matrix based on cross-validation predictions
conf_matrix_rf_cv = confusion_matrix(y_rf, y_rf_cv_pred)
print("\nConfusion Matrix (Cross-Validation):")
print(conf_matrix_rf_cv)


#### Now we train on the whole train dataset and select the top features 

# Exclude specified features
excluded_features = ['num_nodes', 'num_edges', 'in_degree_centrality', 'num_connected_components']
X_rf = train_set_oversampled.drop(columns=excluded_features + ['class'])  # Features
y_rf = train_set_oversampled['class']  # Target variable

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=20, max_depth=25, max_features=60, random_state=1001)
rf_model.fit(X_rf, y_rf)

# Extract feature importance scores
feature_importances = rf_model.feature_importances_
feature_names = X_rf.columns

# Combine feature names and importance scores into a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top features
print("Feature Importance Scores:")
print(feature_importance_df)

# Select the top k features
top_k = 100
selected_features = feature_importance_df.head(top_k)['Feature'].tolist()
print(f"Top {top_k} Features: {selected_features}")

# Create a new dataset with only the selected features
X_selected = X_rf[selected_features]

# Train a new Random Forest model using only the selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1001)
rf_model_selected.fit(X_selected, y_rf)

# Evaluate the new model (e.g., using cross-validation)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model_selected, X_selected, y_rf, cv=5, scoring='f1')
print("5-Fold Cross-Validation Results with Selected Features:")
print(f"Accuracy Scores: {cv_scores}")
print(f"Mean F1-Score: {cv_scores.mean():.4f}")