### This script will adjust the final model to the train data and 
# evaluate the performance of the model on the test data.

#load libraries
import os
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
import joblib
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

top_100_features = ['feature_53', 'feature_55', 'feature_14', 'feature_138', 'feature_49', 'feature_41', 'feature_5', 'feature_132', 'feature_47', 'feature_90', 'feature_29', 'feature_18', 'feature_60', 'feature_66', 'feature_43', 'feature_163', 'feature_2', 'feature_52', 'feature_23', 
'feature_142', 'feature_59', 'feature_46', 'feature_80', 'feature_40', 'feature_25', 'betweenness_centrality', 'feature_28', 'avg_shortest_path_length', 'feature_81', 'feature_31', 'feature_64', 'feature_67', 'feature_100', 'feature_54', 'feature_76', 'feature_61', 'feature_48', 'feature_101', 'feature_136', 'feature_16', 'feature_83', 'feature_65', 'feature_103', 'feature_58', 'feature_84', 'feature_77', 'feature_30', 'feature_139', 'feature_106', 'feature_160', 'feature_156', 'feature_125', 'feature_42', 'feature_85', 'feature_17', 'feature_158', 'feature_137', 'feature_161', 'feature_1', 'feature_21', 'feature_8', 'feature_78', 'feature_22', 'feature_89', 'feature_96', 'feature_91', 'feature_3', 'feature_19', 'feature_9', 'feature_108', 'feature_79', 'feature_127', 'feature_94', 'feature_131', 'feature_159', 'feature_154', 'feature_109', 'feature_155', 'feature_130', 'feature_72', 'feature_144', 'feature_24', 'feature_4', 'feature_114', 'eigenvector_centrality', 'feature_10', 'feature_102', 'feature_97', 'feature_164', 'density', 'feature_95', 'feature_133', 'feature_115', 'avg_degree', 'feature_157', 'feature_145', 'clustering_coefficient_x', 'feature_88', 'feature_71', 'feature_73']

# Define base paths dynamically
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
data_dir = os.path.join(base_dir, "data", "elliptic_bitcoin_dataset")
output_dir = os.path.join(base_dir, "outputs")
rf_model_save_path = os.path.join(output_dir, "random_forest_model.pkl")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File paths
train_set_clean_path = os.path.join(output_dir, "train_set_clean.csv")
test_set_path = os.path.join(output_dir, "test_set.csv")
edgelist_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

# Load the saved Random Forest model
loaded_rf_model = joblib.load(rf_model_save_path)
print("Random Forest model loaded successfully!")


# Load the test set
#test_set_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\test_set.csv'
test_set = pd.read_csv(test_set_path)

# Drop rows where the target variable ('class') is NaN
test_set = test_set.dropna(subset=['class'])

# Separate features and target variable
X_test = test_set[top_100_features] # Adjust columns as needed
y_test = test_set['class']

# Use the trained model to make predictions on the test set
y_test_pred = loaded_rf_model.predict(X_test)

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
pdp_features = ['betweenness_centrality', 'avg_shortest_path_length', 'clustering_coefficient_x', 'eigenvector_centrality', 'density', 'avg_degree']

# Ensure the features exist in the test set
missing_features = [feature for feature in pdp_features if feature not in X_test.columns]
if missing_features:
    print(f"Missing features in test set: {missing_features}")
else:
    # Generate partial dependence plots
    print("Generating Partial Dependence Plots...")
    fig, ax = plt.subplots(figsize=(15, 10))
    PartialDependenceDisplay.from_estimator(
        loaded_rf_model,  # Trained Random Forest model
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
        y_test_pred_subset = loaded_rf_model.predict(X_test_subset)

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
#edgelist_file = r'C:\Users\mario\elliptic-bitcoin-aml\data\elliptic_bitcoin_dataset\elliptic_txs_edgelist.csv'
edgelist_df = pd.read_csv(edgelist_file)
### Prepare the data to run the graph model
# Ensure the edgelist DataFrame has the required columns    


# File path for saving the PDF
graph_visualization_pdf = os.path.join(output_dir, "graph_model_visualization_rf.pdf")
# Create a PdfPages object to save all plots
with PdfPages(graph_visualization_pdf) as pdf:
    for timestamp in sorted(test_set['timestep'].unique()):
        test_subset = test_set[test_set['timestep'] == timestamp]
        if 'transaction_id' not in test_subset.columns:
            print(f"Error: 'transaction_id' column is missing in the test set for timestamp {timestamp}.")
            continue

        transaction_ids = set(test_subset['transaction_id'])
        filtered_edgelist = edgelist_df[
            edgelist_df['txId1'].isin(transaction_ids) | edgelist_df['txId2'].isin(transaction_ids)
        ]

        # Create the graph
        G = nx.from_pandas_edgelist(filtered_edgelist, source='txId1', target='txId2', create_using=nx.DiGraph())
        graph_nodes = list(G.nodes)
        graph_nodes_df = pd.DataFrame({'transaction_id': graph_nodes})
        graph_nodes_df = graph_nodes_df.merge(test_subset, on='transaction_id', how='left')

        # Predict for all nodes in the graph
        X_graph = graph_nodes_df[top_100_features].fillna(0)  # Fill missing features with 0
        y_graph_pred = loaded_rf_model.predict(X_graph)

        # Assign colors based on predictions
        node_colors_pred = ['red' if pred == 1 else 'blue' for pred in y_graph_pred]

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
        axes[0].set_title(f"Predicted Illicit Transactions (Timestamp {timestamp})")

        # Graph with actual illicit transactions
        nx.draw(
            G,
            ax=axes[1],
            with_labels=False,
            node_color=node_colors_actual,
            node_size=50,
            edge_color='gray'
        )
        axes[1].set_title(f"Actual Illicit Transactions (Timestamp {timestamp})")

        # Save the current figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)  # Close the figure to free memory
