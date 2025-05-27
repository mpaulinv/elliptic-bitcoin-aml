### This script will conduct the analysis of the association between the features created in the feature engineering script and available in the dataset, and the target variable.
# # #load libraries
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx


#load data

# Define base paths dynamically
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
data_dir = os.path.join(base_dir, "data", "elliptic_bitcoin_dataset")
output_dir = os.path.join(base_dir, "outputs")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# File paths
classes_file = os.path.join(data_dir, "elliptic_txs_classes.csv")
features_file = os.path.join(data_dir, "elliptic_txs_features.csv")
edgelist_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
timestep_graph_features_path = os.path.join(output_dir, "timestep_graph_features.csv")
transaction_graph_features_path = os.path.join(output_dir, "transaction_graph_features_per_timestep.csv")
ego_network_features_path = os.path.join(output_dir, "ego_network_features_per_timestep.csv")

# Load the data
classes_df = pd.read_csv(classes_file)
features_df = pd.read_csv(features_file, header=None)  # Assuming no header in features
edgelist_df = pd.read_csv(edgelist_file)
timestep_graph_features_df = pd.read_csv(timestep_graph_features_path)
transaction_graph_features_df = pd.read_csv(transaction_graph_features_path)
ego_network_features_df = pd.read_csv(ego_network_features_path)

# Display the first few rows of each DataFrame to verify the import
print("Timestep Graph Features:")
print(timestep_graph_features_df.head(), "\n")

print("Transaction Graph Features:")
print(transaction_graph_features_df.head(), "\n")

print("Ego Network Features:")
print(ego_network_features_df.head(), "\n")

# Rename the columns in the features dataset
features_df.columns = ['transaction_id', 'timestep'] + [f'feature_{i}' for i in range(1, features_df.shape[1] - 1)]


#features_df = features_df.rename(columns={0: 'transaction_id'})
#features_df = features_df.rename(columns={1: 'timestep'})

# Merge features features with ego network features
combined_features_df = pd.merge(features_df, ego_network_features_df, on=['transaction_id', 'timestep'], how='left')

# Merge with timestep-level features
combined_features_df = pd.merge(combined_features_df, timestep_graph_features_df, on='timestep', how='left')

# Merge with transaction-level graph features
combined_features_df = pd.merge(combined_features_df, transaction_graph_features_df, on=['transaction_id', 'timestep'], how='left')

print("Combined features:")
print(timestep_graph_features_df.head(), "\n")

# Save the combined features to a CSV file
combined_features_output_path = os.path.join(output_dir, "combined_features.csv")
combined_features_df.to_csv(combined_features_output_path, index=False)
print(f"Combined graph features saved to: {combined_features_output_path}")

### Now we merge to the combined features the classes_df, which contains the target variable.
# We will use the transaction_id as the key to merge.
# merge classes_df with combined_features_df
classes_df['class'] = classes_df['class'].map({'unknown': np.nan, '1': 1, '2': 0})
classes_df = classes_df.rename(columns={'txId': 'transaction_id'})
final_dataset = pd.merge(combined_features_df, classes_df, on='transaction_id', how='left')


# Split the dataset into train and test sets based on the timestep column. We will keep the last 15 timesteps for testing 
# and the rest for training. This is a common practice in time series analysis to ensure that the model is trained on past data and tested on future data.

train_set = final_dataset[final_dataset['timestep'] <= 34]  # First 34 timesteps for training
test_set = final_dataset[final_dataset['timestep'] > 34]    # Remaining timesteps for testing

### Now we will produce some exploratory analysis of the features and the target variable.
### First, we just want to keep the observations that have a target variable, so we will drop the rows with NaN in the target variable.
train_set_clean = train_set.dropna(subset=['class'])

print("Data types of train_set_clean:")
print(train_set_clean.dtypes)

target_variable = 'class'
features = [col for col in train_set_clean.columns if col not in ['transaction_id', 'timestep', target_variable]]

# Create scatterplots for each feature against the target variable
scatterplots_with_regression_pdf = os.path.join(output_dir, "scatterplots_with_regression.pdf")
with PdfPages(scatterplots_with_regression_pdf) as pdf:
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=train_set_clean,
            x=feature,
            y=target_variable,
            scatter_kws={'alpha': 0.5},  # Transparency for scatter points
            line_kws={'color': 'red'},  # Regression line color
            ci=None  # Disable confidence interval for clarity
        )
        plt.title(f'Scatterplot of {target_variable} vs {feature}')
        plt.xlabel(feature)
        plt.ylabel(target_variable)
        plt.grid(True)
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the figure to avoid overlapping plots

print(f"Scatterplots with regression lines saved to: {scatterplots_with_regression_pdf}")


# Exclude columns starting with 'feature_' from the features list
filtered_features = [
    col for col in train_set_clean.columns
    if col not in ['transaction_id', 'timestep', target_variable] and not col.startswith('feature_')
]

# Verify the filtered features
print("Filtered features (excluding 'feature_'):")
print(filtered_features)

# Create scatterplots for each feature against the target variable
scatterplots_without_features_pdf = os.path.join(output_dir, "scatterplots_without_feature_columns.pdf")
with PdfPages(scatterplots_without_features_pdf) as pdf:
    for feature in filtered_features:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=train_set_clean,
            x=feature,
            y=target_variable,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
            ci=None
        )
        plt.title(f'Scatterplot of {target_variable} vs {feature}')
        plt.xlabel(feature)
        plt.ylabel(target_variable)
        plt.grid(True)
        pdf.savefig()
        plt.close()

print(f"Scatterplots without 'feature_' columns saved to: {scatterplots_without_features_pdf}")

# Filter the columns for the heatmap
heatmap_columns = [
    col for col in train_set_clean.columns
    if col not in ['transaction_id', 'timestep'] and not col.startswith('feature_')
]

# Compute the correlation matrix
correlation_matrix = train_set_clean[heatmap_columns].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Display correlation values
    fmt=".2f",   # Format for the correlation values
    cmap="coolwarm",  # Color map
    cbar=True,  # Show color bar
    square=True  # Make the heatmap square
)
plt.title("Heatmap Correlogram of Selected Features")
plt.tight_layout()

# Save the heatmap to a file
heatmap_output_path = os.path.join(output_dir, "heatmap_correlogram_graph_features.png")
plt.savefig(heatmap_output_path)
plt.show()

print(f"Heatmap correlogram saved to: {heatmap_output_path}")

# Save train and test sets to CSV files
train_set_clean_output_path = os.path.join(output_dir, "train_set_clean.csv")
test_set_output_path = os.path.join(output_dir, "test_set.csv")
train_set_clean.to_csv(train_set_clean_output_path, index=False)
test_set.to_csv(test_set_output_path, index=False)

print(f"Train set saved to: {train_set_clean_output_path}")
print(f"Test set saved to: {test_set_output_path}")


### Comments: Some of the premade features are highly correlated with each other. These could be removed from the model. 
### Regarding the target variable, we can see that the dataset is highly imbalanced. 
### As per the graph features we would expect people engaging in illicit activities would tend to form closely connented groups,
### at the edges of the graph. We would so expect them to show related transactions with many nodes during layering and 
### with many transactions with the same node during integration.

## In the scatterplots we see. Ego size: inversely related to target
## Ego density: if there is a relationship is not linear. Maybe middle values are more relevant here than low or high values 
## Avg neigbor: slight positive 
## number of nodes, edges and degree slightly negative
## density is positive while clustering negative 
## Degree centrality is negative and intuitively so is pagerank. 

