## Feature engineering. This script creates features for the prediction of 
## ilicit activity in the elliptic dataset.
## The features are based on the graph structure of the transactions 

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


### We can create some features based on the general graph structure on a given time step, as we saw in 
### the EDA notebook. We can create the following features:

# Create a directed graph from the edgelist
G = nx.from_pandas_edgelist(edgelist_df, source='txId1', target='txId2', create_using=nx.DiGraph())

# Initialize lists to store results
timestep_features = []

# Loop through each timestep
for timestep in sorted(features_df[1].unique()):  # Assuming column 1 in features_df is 'timestep'
    # Get the nodes for the current timestep
    nodes_in_timestep = features_df[features_df[1] == timestep][0]  # Assuming column 0 is 'transaction_id'
    
    # Subgraph for the current timestep
    subgraph = G.subgraph(nodes_in_timestep)
    
    # Calculate general graph features
    num_nodes = len(subgraph.nodes)
    num_edges = len(subgraph.edges)
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    density = nx.density(subgraph) if num_nodes > 1 else 0
    clustering_coefficient = nx.average_clustering(subgraph.to_undirected()) if num_nodes > 1 else 0
    num_connected_components = nx.number_connected_components(subgraph.to_undirected()) if num_nodes > 0 else 0
    
    # Append the features for this timestep
    timestep_features.append({
        'timestep': timestep,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'density': density,
        'clustering_coefficient': clustering_coefficient,
        'num_connected_components': num_connected_components
    })

# Convert the results to a DataFrame
timestep_features_df = pd.DataFrame(timestep_features)

# Display the features
print(timestep_features_df.head())

# Save the features to a CSV file (optional)
output_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\timestep_graph_features.csv'
timestep_features_df.to_csv(output_path, index=False)
print(f"Timestep graph features saved to: {output_path}")

## now we can also create some features based on the graph structure around the transactions 


# Calculate transaction-level graph features
# Initialize dictionaries to store features
# Initialize a list to store transaction-level features for all timesteps
transaction_features = []

# Loop through each timestep
for timestep in sorted(features_df[1].unique()):  # Assuming column 1 in features_df is 'timestep'
    print(f"Processing timestep {timestep}...")
    
    # Get the nodes for the current timestep
    nodes_in_timestep = features_df[features_df[1] == timestep][0]  # Assuming column 0 is 'transaction_id'
    
    # Subgraph for the current timestep
    subgraph = G.subgraph(nodes_in_timestep)
    
    # Calculate graph features for the subgraph
    degree_centrality = dict(subgraph.degree())  # Total degree (in + out)
    in_degree_centrality = dict(subgraph.in_degree())  # In-degree
    out_degree_centrality = dict(subgraph.out_degree())  # Out-degree
    clustering_coefficient = nx.clustering(subgraph.to_undirected())  # Clustering coefficient
    betweenness_centrality = nx.betweenness_centrality(subgraph)  # Betweenness centrality
    pagerank = nx.pagerank(subgraph)  # PageRank
    
    # Store features for each transaction in the current timestep
    for node in subgraph.nodes:
        transaction_features.append({
            'transaction_id': node,
            'timestep': timestep,
            'degree_centrality': degree_centrality.get(node, 0),
            'in_degree_centrality': in_degree_centrality.get(node, 0),
            'out_degree_centrality': out_degree_centrality.get(node, 0),
            'clustering_coefficient': clustering_coefficient.get(node, 0),
            'betweenness_centrality': betweenness_centrality.get(node, 0),
            'pagerank': pagerank.get(node, 0)
        })

# Convert the results to a DataFrame
transaction_features_df = pd.DataFrame(transaction_features)

# Display the features
print(transaction_features_df.head())

# Save the transaction-level features to a CSV file (optional)
transaction_features_output_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\transaction_graph_features_per_timestep.csv'
transaction_features_df.to_csv(transaction_features_output_path, index=False)
print(f"Transaction-level graph features saved to: {transaction_features_output_path}")


# Initialize a list to store ego network features for all transactions
ego_features = []

# Loop through each timestep
for timestep in sorted(features_df[1].unique()):  # Assuming column 1 in features_df is 'timestep'
    print(f"Processing ego features for timestep {timestep}...")
    
    # Get the nodes for the current timestep
    nodes_in_timestep = features_df[features_df[1] == timestep][0]  # Assuming column 0 is 'transaction_id'
    
    # Subgraph for the current timestep
    subgraph = G.subgraph(nodes_in_timestep)
    
    # Calculate ego network features for each node in the subgraph
    for node in subgraph.nodes:
        # Construct the ego network for the node
        ego_network = nx.ego_graph(subgraph, node)
        
        # Calculate ego network features
        ego_size = len(ego_network.nodes)  # Ego network size
        ego_density = nx.density(ego_network)  # Ego network density
        neighbor_degrees = [subgraph.degree(neighbor) for neighbor in ego_network.neighbors(node)]
        avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees) if neighbor_degrees else 0
        
        # Append the features for this node
        ego_features.append({
            'transaction_id': node,
            'timestep': timestep,
            'ego_size': ego_size,
            'ego_density': ego_density,
            'avg_neighbor_degree': avg_neighbor_degree
        })

# Convert the results to a DataFrame
ego_features_df = pd.DataFrame(ego_features)

# Display the features
print(ego_features_df.head())

# Save the ego network features to a CSV file
ego_features_output_path = r'C:\Users\mario\elliptic-bitcoin-aml\outputs\ego_network_features_per_timestep.csv'
ego_features_df.to_csv(ego_features_output_path, index=False)
print(f"Ego network features saved to: {ego_features_output_path}")

