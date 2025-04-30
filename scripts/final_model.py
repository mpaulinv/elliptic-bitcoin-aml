### This script will adjust the final model to the train data and 
# evaluate the performance of the model on the test data.

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
from sklearn.inspection import PartialDependenceDisplay

top_100_features = ['feature_55', 'feature_53', 'feature_90', 'feature_14', 'feature_18', 'feature_132', 'feature_47', 'feature_163', 'betweenness_centrality', 'feature_5', 'feature_43', 'feature_46', 'feature_49', 'feature_2', 'feature_103', 'feature_65', 'feature_80', 'feature_41', 'feature_100', 'feature_59', 'feature_136', 'feature_48', 'feature_29', 'feature_52', 'feature_81', 'feature_67', 'feature_101', 'feature_8', 'feature_107', 'feature_1', 'feature_23', 'feature_16', 'feature_60', 'feature_139', 'feature_160', 'feature_137', 'feature_77', 'feature_58', 'feature_85', 'feature_40', 'feature_31', 'feature_142', 'feature_94', 'feature_61', 'feature_106', 'feature_89', 'feature_3', 'feature_64', 'feature_127', 'feature_17', 'feature_154', 'feature_138', 'feature_159', 'feature_11', 'feature_25', 'feature_95', 'feature_91', 'feature_9', 'feature_131', 'feature_125', 'feature_66', 'feature_130', 'feature_72', 'avg_degree', 'feature_84', 'clustering_coefficient_x', 'feature_88', 'feature_155', 'feature_28', 'feature_102', 'feature_144', 'feature_165', 'feature_143', 'feature_71', 'feature_164', 'feature_79', 'feature_151', 'feature_42', 'feature_157', 'feature_158', 'feature_121', 'feature_73', 'feature_145', 'feature_19', 'feature_24', 'pagerank', 'feature_115', 'feature_12', 'feature_97', 'feature_133', 'feature_109', 'feature_114', 'density', 'feature_104', 'feature_156', 'feature_82', 'feature_148', 'feature_96', 'feature_22', 'feature_6']
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

### Train the model on the oversampled dataset
X_rf = train_set_oversampled[top_100_features]  # Features
y_rf = train_set_oversampled['class']  # Target variable

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=20, max_depth=25, max_features=60, random_state=1001)
rf_model.fit(X_rf, y_rf)

# Load the test set
test_set_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\test_set.csv'
test_set = pd.read_csv(test_set_path)

# Drop rows where the target variable ('class') is NaN
test_set = test_set.dropna(subset=['class'])

# Separate features and target variable
X_test = test_set[top_100_features] # Adjust columns as needed
y_test = test_set['class']

# Use the trained model to make predictions on the test set
y_test_pred = rf_model.predict(X_test)

# Generate a classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred, target_names=["Legitimate (0)", "Illicit (1)"]))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix on Test Set:")
print(conf_matrix)


### top features of this study 'betweenness_centrality'
# 'avg_degree'
# 'clustering_coefficient_x'
# 'pagerank'
# 'density'

### Partial development plots of the features 

# List of features for partial dependence plots
pdp_features = ['betweenness_centrality', 'avg_degree', 'clustering_coefficient_x', 'pagerank', 'density']

# Ensure the features exist in the test set
missing_features = [feature for feature in pdp_features if feature not in X_test.columns]
if missing_features:
    print(f"Missing features in test set: {missing_features}")
else:
    # Generate partial dependence plots
    print("Generating Partial Dependence Plots...")
    fig, ax = plt.subplots(figsize=(15, 10))
    PartialDependenceDisplay.from_estimator(
        rf_model,  # Trained Random Forest model
        X_test,    # Test set features
        pdp_features,  # Features to plot
        ax=ax
    )
    plt.tight_layout()
    plt.show()

# Ensure the 'timestep' column exists in the test set
if 'timestep' not in test_set.columns:
    print("Error: 'timestep' column is missing in the test set.")
else:
    # Group the test set by timestep
    f1_scores = []
    timesteps = sorted(test_set['timestep'].unique())

    for timestep in timesteps:
        # Filter the test set for the current timestep
        test_subset = test_set[test_set['timestep'] == timestep]
        X_test_subset = test_subset[top_100_features]
        y_test_subset = test_subset['class']

        # Make predictions for the current timestep
        y_test_pred_subset = rf_model.predict(X_test_subset)

        # Calculate the F1-score for illicit transactions (class 1)
        f1 = f1_score(y_test_subset, y_test_pred_subset, pos_label=1)
        f1_scores.append(f1)

    # Plot the F1-score against the timestep
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, f1_scores, marker='o', linestyle='-', color='b')
    plt.title("F1-Score for Illicit Transactions (Class 1) Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.show()

# Load the edgelist data
# Load the edgelist data
edgelist_file = r'C:\Users\mario\elliptic-bitcoin-aml\data\elliptic_bitcoin_dataset\elliptic_txs_edgelist.csv'
edgelist_df = pd.read_csv(edgelist_file)

# Filter the test set for the specific timestamp
timestamp = 37
test_subset = test_set[test_set['timestep'] == timestamp]

# Ensure the necessary columns exist
if 'transaction_id' not in test_subset.columns:
    print("Error: 'transaction_id' column is missing in the test set.")
else:
    # Filter the edgelist for the transactions in the current timestamp
    transaction_ids = set(test_subset['transaction_id'])
    filtered_edgelist = edgelist_df[
        edgelist_df['txId1'].isin(transaction_ids) | edgelist_df['txId2'].isin(transaction_ids)
    ]

    # Create the graph
    G = nx.from_pandas_edgelist(filtered_edgelist, source='txId1', target='txId2', create_using=nx.DiGraph())

    # Get all nodes in the graph
    graph_nodes = list(G.nodes)

    # Prepare a DataFrame for all nodes in the graph
    graph_nodes_df = pd.DataFrame({'transaction_id': graph_nodes})

    # Merge with the test subset to include features and labels for known nodes
    graph_nodes_df = graph_nodes_df.merge(test_subset, on='transaction_id', how='left')

    # Predict for all nodes in the graph
    X_graph = graph_nodes_df[top_100_features].fillna(0)  # Fill missing features with 0
    y_graph_pred = rf_model.predict(X_graph)

    # Assign colors based on predictions
    node_colors_pred = [
        'red' if pred == 1 else 'blue' for pred in y_graph_pred
    ]

    # Assign colors based on actual labels (gray for unknown labels)
    node_colors_actual = [
        'red' if actual == 1 else 'blue' if actual == 0 else 'gray'
        for actual in graph_nodes_df['class']
    ]

    # Plot the graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Graph with predicted illicit transactions
    nx.draw(
        G,
        ax=axes[0],
        with_labels=False,
        node_color=node_colors_pred,
        node_size=50,
        edge_color='gray'
    )
    axes[0].set_title("Predicted Illicit Transactions")

    # Graph with actual illicit transactions
    nx.draw(
        G,
        ax=axes[1],
        with_labels=False,
        node_color=node_colors_actual,
        node_size=50,
        edge_color='gray'
    )
    axes[1].set_title("Actual Illicit Transactions")

    # Show the plots
    plt.tight_layout()
    plt.show()





# Filter the test set for the specific timestamp
timestamp = 49
test_subset = test_set[test_set['timestep'] == timestamp]

# Ensure the necessary columns exist
if 'transaction_id' not in test_subset.columns:
    print("Error: 'transaction_id' column is missing in the test set.")
else:
    # Filter the edgelist for the transactions in the current timestamp
    transaction_ids = set(test_subset['transaction_id'])
    filtered_edgelist = edgelist_df[
        edgelist_df['txId1'].isin(transaction_ids) | edgelist_df['txId2'].isin(transaction_ids)
    ]

    # Create the graph
    G = nx.from_pandas_edgelist(filtered_edgelist, source='txId1', target='txId2', create_using=nx.DiGraph())

    # Get all nodes in the graph
    graph_nodes = list(G.nodes)

    # Prepare a DataFrame for all nodes in the graph
    graph_nodes_df = pd.DataFrame({'transaction_id': graph_nodes})

    # Merge with the test subset to include features and labels for known nodes
    graph_nodes_df = graph_nodes_df.merge(test_subset, on='transaction_id', how='left')

    # Predict for all nodes in the graph
    X_graph = graph_nodes_df[top_100_features].fillna(0)  # Fill missing features with 0
    y_graph_pred = rf_model.predict(X_graph)

    # Assign colors based on predictions
    node_colors_pred = [
        'red' if pred == 1 else 'blue' for pred in y_graph_pred
    ]

    # Assign colors based on actual labels (gray for unknown labels)
    node_colors_actual = [
        'red' if actual == 1 else 'blue' if actual == 0 else 'gray'
        for actual in graph_nodes_df['class']
    ]

    # Plot the graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Graph with predicted illicit transactions
    nx.draw(
        G,
        ax=axes[0],
        with_labels=False,
        node_color=node_colors_pred,
        node_size=50,
        edge_color='gray'
    )
    axes[0].set_title("Predicted Illicit Transactions")

    # Graph with actual illicit transactions
    nx.draw(
        G,
        ax=axes[1],
        with_labels=False,
        node_color=node_colors_actual,
        node_size=50,
        edge_color='gray'
    )
    axes[1].set_title("Actual Illicit Transactions")

    # Show the plots
    plt.tight_layout()
    plt.show()