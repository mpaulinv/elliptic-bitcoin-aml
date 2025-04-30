#exploratory data analysis 
# This script performs exploratory data analysis (EDA) on the Elliptic Bitcoin dataset.
# It includes loading the dataset, checking for missing values, visualizing distributions, and analyzing correlations.

#load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx

#load data
# File paths
base_path = r'C:\Users\mario\elliptic-bitcoin-aml\data\elliptic_bitcoin_dataset'
classes_file = f'{base_path}/elliptic_txs_classes.csv'
features_file = f'{base_path}/elliptic_txs_features.csv'
edgelist_file = f'{base_path}/elliptic_txs_edgelist.csv'

# Load the data
classes_df = pd.read_csv(classes_file)
features_df = pd.read_csv(features_file, header=None)  # Assuming no header in features
edgelist_df = pd.read_csv(edgelist_file)

# Display basic info about each dataset
print("Classes Dataset:")
print(classes_df.head(), "\n")
print("Features Dataset:")
print(features_df.head(), "\n")
print("Edgelist Dataset:")
print(edgelist_df.head(), "\n")

#Classes dataset: Contains the labels for each transaction. Some transactions are labeled as "unknown" (0), while others are labeled as "legitimate" (1) or "illicit" (2).
## Exploratory analysis of the classes dataset 

# Check for missing values
print("\nMissing Values in Classes Dataset:")
print(classes_df.isnull().sum())

# Check the distribution of classes
print("\nClass Distribution:")
print(classes_df['class'].value_counts())  # Replace 'class' with the actual column name

# Visualize the class distribution
sns.countplot(data=classes_df, x='class')  # Replace 'class' with the actual column name
plt.title("Class Distribution")
plt.show()

#157205 transactions labeled as uknown, 4545 as ilicit and 42019 as legitimate. May need to oversample the ilicit to balance the dataset.

# EDA for the features dataset

# Rename the columns in the features dataset
features_df.columns = ['transaction_id', 'timestep'] + [f'feature_{i}' for i in range(1, features_df.shape[1] - 1)]

# Display the first few rows to verify the changes
print("Renamed Features Dataset:")
print(features_df.head())

# Check the shape of the dataset
print("\nFeatures Dataset Shape:", features_df.shape)

# Display summary statistics
print("\nSummary Statistics for Features Dataset:")
print(features_df.describe())

# Check for missing values
print("\nMissing Values in Features Dataset:")
print(features_df.isnull().sum())

# Visualize the distribution of a few features
features_df.iloc[:, 1:5].hist(bins=30, figsize=(10, 8))  # Visualize the first few features
plt.suptitle("Feature Distributions")
plt.show()

# File paths for saving CSVs
missing_values_csv = r"C:\Users\mario\elliptic-bitcoin-aml\outputs\missing_values_features.csv"
summary_statistics_csv = r"C:\Users\mario\elliptic-bitcoin-aml\outputs\summary_statistics_features.csv"
feature_histograms_pdf = r"C:\Users\mario\elliptic-bitcoin-aml\outputs\feature_histograms.pdf"

# Save Missing Values to a CSV
missing_values = features_df.isnull().sum()
missing_values.to_csv(missing_values_csv, header=["Missing Values"], index_label="Feature")
print(f"Missing values saved to: {missing_values_csv}")

# Save Summary Statistics to a CSV
summary_stats = features_df.describe()
summary_stats.to_csv(summary_statistics_csv)
print(f"Summary statistics saved to: {summary_statistics_csv}")

# Save Histograms for All Features to a PDF
with PdfPages(feature_histograms_pdf) as pdf:
    for col in features_df.columns[1:]:  # Assuming the first column is transaction ID
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(features_df[col], bins=30, kde=False, ax=ax)
        ax.set_title(f"Histogram for Feature {col}")
        ax.set_xlabel(f"Feature {col}")
        ax.set_ylabel("Frequency")
        pdf.savefig(fig)
        plt.close(fig)

print(f"Feature histograms saved to: {feature_histograms_pdf}")

### As per the paper accompanying the dataset, the features are:
### 1-94 local features including the time step, number of inputs/outputs, transaction fee, output
### volume and aggregated figures such as average BTC received (spent)
### by the inputs/outputs and average number of incoming (outgoing)
### transactions associated with the inputs/outputs

### 95-166 global features aggregated features, are obtained by aggregating
## transaction information one-hop backward/forward from the center node - giving the maximum, minimum, standard deviation and
### correlation coefficients of the neighbour transactions for the same
## information data

## No missing values in the features dataset.

### From the inspection of the summary statistics we can see that the features seem to be stantardized 
### with mean 0 and standard deviation 1. They tend to have some high positive maximum values 
### not large negative values. 

# Question, are the features ortogonal? We can create a correlation matrix to check this.

# Compute the correlation matrix

# Compute the correlation matrix
correlation_matrix = features_df.iloc[:, 1:].corr()  # Exclude the first column if it's a transaction ID

# Plot the heatmap
plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Features")
plt.show()

# Save the correlation matrix to a CSV
correlation_matrix_csv = r"C:\Users\mario\elliptic-bitcoin-aml\outputs\correlation_matrix_features.csv"
correlation_matrix.to_csv(correlation_matrix_csv)
print(f"Correlation matrix saved to: {correlation_matrix_csv}")

# Find pairs of features with correlation > 0.80 or < -0.80
high_correlation_pairs = (
    correlation_matrix
    .where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))  # Keep only upper triangle (no duplicates)
    .stack()  # Convert to long format
    .reset_index()  # Reset index for easier filtering
)
high_correlation_pairs.columns = ['Feature1', 'Feature2', 'Correlation']

# Filter pairs with correlation > 0.80 or < -0.80
high_correlation_pairs = high_correlation_pairs[
    high_correlation_pairs['Correlation'].abs() > 0.80
]

# Display the results
print("Highly Correlated Pairs (|correlation| > 0.80):")
print(high_correlation_pairs)

# Save the results to a CSV for further inspection
high_correlation_csv = r"C:\Users\mario\elliptic-bitcoin-aml\outputs\high_correlation_pairs.csv"
high_correlation_pairs.to_csv(high_correlation_csv, index=False)
print(f"Highly correlated pairs saved to: {high_correlation_csv}")

### Analysis of the data across time steps

# Count the number of transactions per time step
transactions_per_timestep = features_df['timestep'].value_counts().sort_index()

# Plot the number of transactions per time step
plt.figure(figsize=(10, 6))
sns.barplot(x=transactions_per_timestep.index, y=transactions_per_timestep.values, color='skyblue')
plt.title("Number of Transactions per Time Step")
plt.xlabel("Time Step")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Plot the transactions per class by timestep 


# Merge the features and classes datasets on transaction_id
merged_df = pd.merge(features_df, classes_df, left_on='transaction_id', right_on='txId')

# Group by timestep and class to count the number of transactions per class in each timestep
class_counts_per_timestep = merged_df.groupby(['timestep', 'class']).size().unstack(fill_value=0)

# Calculate the total transactions per timestep
total_transactions_per_timestep = class_counts_per_timestep.sum(axis=1)

# Calculate the share of each class per timestep
class_share_per_timestep = class_counts_per_timestep.div(total_transactions_per_timestep, axis=0)

# Plot the share of each class over time
plt.figure(figsize=(12, 6))
for class_label in class_share_per_timestep.columns:
    plt.plot(class_share_per_timestep.index, class_share_per_timestep[class_label], label=f"Class {class_label}")

plt.title("Share of Transaction Classes per Time Step")
plt.xlabel("Time Step")
plt.ylabel("Share of Transactions")
plt.legend(title="Class", labels=["Illicit (0)", "Licit (1)", "Uknown (2)"])
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate the ratio of illicit (class = 0) to licit (class = 1) transactions per time step
# Ensure the 'class' column is numeric

print("Unique values in 'class' column:")
print(classes_df['class'].unique())

print("class_counts_per_timestep DataFrame:")
print(class_counts_per_timestep.head())

print("Data type of 'class' column in classes_df:")
print(classes_df['class'].dtype)

print("Data types of columns in class_counts_per_timestep:")
print(class_counts_per_timestep.dtypes)


# Replace 'unknown' with 0 and convert the 'class' column to integers
classes_df['class'] = classes_df['class'].replace('unknown', 0).astype(int)

# Verify the unique values in the 'class' column
print("Unique values in 'class' column after cleaning:")
print(classes_df['class'].unique())

# Merge the features and classes datasets on transaction_id
merged_df = pd.merge(features_df, classes_df, left_on='transaction_id', right_on='txId')

# Group by timestep and class to count the number of transactions per class in each timestep
class_counts_per_timestep = merged_df.groupby(['timestep', 'class']).size().unstack(fill_value=0)

# Ensure column names are integers
class_counts_per_timestep.columns = class_counts_per_timestep.columns.astype(int)

# Verify the cleaned class_counts_per_timestep DataFrame
print("class_counts_per_timestep DataFrame after cleaning:")
print(class_counts_per_timestep.head())

# Calculate the ratio of illicit (class = 0) to licit (class = 1) transactions per time step
illicit_to_licit_ratio = class_counts_per_timestep[1] / class_counts_per_timestep[2]

# Plot the ratio over time
plt.figure(figsize=(12, 6))
plt.plot(illicit_to_licit_ratio.index, illicit_to_licit_ratio.values, marker='o', color='red', label="Illicit to Licit Ratio")
plt.title("Ratio of Illicit to Licit Transactions Over Time")
plt.xlabel("Time Step")
plt.ylabel("Illicit to Licit Ratio")
plt.axhline(y=1, color='gray', linestyle='--', label="Equal Illicit and Licit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### EDA on the graph structure

print(edgelist_df.head()) 

# Create a directed graph from the edgelist
G = nx.from_pandas_edgelist(edgelist_df, source='txId1', target='txId2', create_using=nx.DiGraph())

# Total number of nodes and edges
total_nodes = G.number_of_nodes()
total_edges = G.number_of_edges()

print(f"Total number of nodes: {total_nodes}")
print(f"Total number of edges: {total_edges}")

# Analysis by timestep 

# Initialize lists to store results
nodes_per_timestep = []
edges_per_timestep = []

# Loop through each timestep
for timestep in sorted(features_df['timestep'].unique()):
    # Get the nodes for the current timestep
    nodes_in_timestep = features_df[features_df['timestep'] == timestep]['transaction_id']
    
    # Subgraph for the current timestep
    subgraph = G.subgraph(nodes_in_timestep)
    
    # Count nodes and edges
    nodes_per_timestep.append(len(subgraph.nodes))
    edges_per_timestep.append(len(subgraph.edges))

# Create a DataFrame to store the results
timestep_graph_stats = pd.DataFrame({
    'Time Step': sorted(features_df['timestep'].unique()),
    'Nodes': nodes_per_timestep,
    'Edges': edges_per_timestep
})

print(timestep_graph_stats)


# Calculate the average degree for each timestep
timestep_graph_stats['Average Degree'] = timestep_graph_stats['Edges'] / timestep_graph_stats['Nodes']

# Display the results
print(timestep_graph_stats[['Time Step', 'Average Degree']])

# Plot the Node-to-Edge Ratio over time
plt.figure(figsize=(10, 6))
plt.plot(timestep_graph_stats['Time Step'], timestep_graph_stats['Average Degree'], marker='o', label="Average Degree")
plt.title("Average Degree Over Time")
plt.xlabel("Time Step")
plt.ylabel("Average Degree")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Plot the Average Degree and Illicit-to-Licit Ratio over time
plt.figure(figsize=(12, 6))

# Plot Average Degree
plt.plot(timestep_graph_stats['Time Step'], timestep_graph_stats['Average Degree'], marker='o', label="Average Degree", color='blue')

# Plot Illicit-to-Licit Ratio on a secondary y-axis
ax1 = plt.gca()  # Get the current axis
ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
ax2.plot(illicit_to_licit_ratio.index, illicit_to_licit_ratio.values, marker='o', label="Illicit to Licit Ratio", color='red')

# Add titles and labels
plt.title("Average Degree and Illicit-to-Licit Ratio Over Time")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Average Degree", color='blue')
ax2.set_ylabel("Illicit to Licit Ratio", color='red')

# Add grid and legends
ax1.grid(True)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()