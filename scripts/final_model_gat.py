### This script evaluates the best hypertuned GAT model (More Attention Heads configuration) on the test data.

# Load libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm, GATv2Conv

# Define base paths dynamically
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
data_dir = os.path.join(base_dir, "data", "elliptic_bitcoin_dataset")
output_dir = os.path.join(base_dir, "outputs")

# Updated model path for the best hypertuned model
gat_model_save_path = os.path.join(output_dir, "best_hypertuned_gat_model.pth")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File paths
test_set_path = os.path.join(output_dir, "test_set.csv")
edgelist_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

# Load datasets
test_set = pd.read_csv(test_set_path)

# Try to load the balanced training data first, fall back to original if not available
balanced_train_path = os.path.join(output_dir, "train_set_balanced_70_30.csv")
original_train_path = os.path.join(output_dir, "train_set_clean.csv")

if os.path.exists(balanced_train_path):
    print("Using balanced training data for feature scaling...")
    train_set_for_scaling = pd.read_csv(balanced_train_path)
    print(f"Balanced training set shape: {train_set_for_scaling.shape}")
    
    # Print class distribution of balanced training data
    class_dist = train_set_for_scaling['class'].value_counts().sort_index()
    print(f"Training data class distribution:")
    print(f"  Licit (0): {class_dist[0]} ({class_dist[0]/len(train_set_for_scaling)*100:.1f}%)")
    print(f"  Illicit (1): {class_dist[1]} ({class_dist[1]/len(train_set_for_scaling)*100:.1f}%)")
else:
    print("Balanced training data not found, using original training data for feature scaling...")
    train_set_for_scaling = pd.read_csv(original_train_path)

edgelist_df = pd.read_csv(edgelist_file)

# Drop rows where the target variable ('class') is NaN
test_set = test_set.dropna(subset=['class'])

print(f"Test set shape: {test_set.shape}")
test_class_dist = test_set['class'].value_counts().sort_index()
print(f"Test data class distribution:")
print(f"  Licit (0): {test_class_dist[0]} ({test_class_dist[0]/len(test_set)*100:.1f}%)")
print(f"  Illicit (1): {test_class_dist[1]} ({test_class_dist[1]/len(test_set)*100:.1f}%)")

# Step 1: Create a mapping from transaction_id to timestep using only the test set
transaction_to_timestep = test_set.set_index('transaction_id')['timestep'].to_dict()

# Step 2: Add timestep1 and timestep2 to edgelist_df
edgelist_df['timestep1'] = edgelist_df['txId1'].map(transaction_to_timestep)
edgelist_df['timestep2'] = edgelist_df['txId2'].map(transaction_to_timestep)

# Step 3: Drop rows where either timestep1 or timestep2 is NaN
edgelist_df = edgelist_df.dropna(subset=['timestep1', 'timestep2'])

# Step 4: Map transaction IDs to consecutive indices using only the test set
node_id_mapping = {node_id: idx for idx, node_id in enumerate(test_set['transaction_id'].unique())}

# Apply the mapping to the edgelist
edgelist_df['txId1_mapped'] = edgelist_df['txId1'].map(node_id_mapping)
edgelist_df['txId2_mapped'] = edgelist_df['txId2'].map(node_id_mapping)

# Drop rows with NaN values in the mapped columns
edgelist_df = edgelist_df.dropna(subset=['txId1_mapped', 'txId2_mapped'])
edgelist_df[['txId1_mapped', 'txId2_mapped']] = edgelist_df[['txId1_mapped', 'txId2_mapped']].astype(int)

# Convert edgelist to PyTorch Geometric format
edge_index = torch.tensor(edgelist_df[['txId1_mapped', 'txId2_mapped']].values.T, dtype=torch.long)

# CRITICAL: Apply the same feature scaling used during training
from sklearn.preprocessing import StandardScaler

# Get feature columns (same as training)
feature_columns = [col for col in test_set.columns if col not in ['class', 'transaction_id', 'timestep']]
train_feature_columns = [col for col in train_set_for_scaling.columns if col not in ['class', 'transaction_id', 'timestep']]

# Ensure feature columns match between train and test
assert feature_columns == train_feature_columns, "Feature columns don't match between train and test sets"

# Fit scaler on training data and transform test data
scaler = StandardScaler()
scaler.fit(train_set_for_scaling[feature_columns])  # Fit on training data (balanced if available)
test_features_scaled = scaler.transform(test_set[feature_columns])  # Transform test data

# Prepare node features and labels from the test set
node_features = torch.tensor(test_features_scaled, dtype=torch.float)
labels = torch.tensor(test_set['class'].values, dtype=torch.long)

# Add self-loops to edge_index (same as training)
num_nodes = len(node_features)
self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
edge_index = torch.cat([edge_index, self_loops], dim=1)

# Create PyTorch Geometric Data object for the test set
test_data = Data(x=node_features, edge_index=edge_index, y=labels)

# Define the GAT model (matching the best hypertuned architecture)
class AdvancedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, num_layers=3, 
                 dropout=0.25, attention_dropout=0.125, use_gatv2=True, residual=True):
        super(AdvancedGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()
        
        # Use GATv2
        conv_layer = GATv2Conv
        
        # First layer
        self.convs.append(conv_layer(
            input_dim, hidden_dim, heads=heads, dropout=attention_dropout, concat=True
        ))
        self.batch_norms.append(BatchNorm(hidden_dim * heads))
        if self.residual:
            self.residual_lins.append(torch.nn.Linear(input_dim, hidden_dim * heads))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_dim * heads, hidden_dim, heads=heads, dropout=attention_dropout, concat=True
            ))
            self.batch_norms.append(BatchNorm(hidden_dim * heads))
            if self.residual:
                self.residual_lins.append(torch.nn.Linear(hidden_dim * heads, hidden_dim * heads))
        
        # Output layer
        self.convs.append(conv_layer(
            hidden_dim * heads, output_dim, heads=1, dropout=attention_dropout, concat=False
        ))
        
        # Additional processing layers
        self.final_dropout = torch.nn.Dropout(dropout)
        self.final_linear = torch.nn.Linear(output_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers - 1):
            x_input = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            
            # Residual connection
            if self.residual and hasattr(self, 'residual_lins'):
                x = x + self.residual_lins[i](x_input)
            
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.final_dropout(x)
        x = self.final_linear(x)
        
        return x

# Best model parameters from hyperparameter tuning
# BEST CONFIGURATION: More Attention Heads
# Best F1-Score: 0.8816
input_dim = node_features.shape[1]
hidden_dim = 12      # Updated from hypertuning
output_dim = 2       # Two neurons for binary classification
heads = 8           # Updated from hypertuning
dropout = 0.4       # Updated from hypertuning
num_layers = 2      # Updated from hypertuning
attention_dropout = dropout * 0.5  # Same calculation as training

print(f"\nBest Model Configuration (More Attention Heads):")
print(f"  Input dim: {input_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Output dim: {output_dim}")
print(f"  Heads: {heads}")
print(f"  Dropout: {dropout}")
print(f"  Attention dropout: {attention_dropout}")
print(f"  Num layers: {num_layers}")
print(f"  Expected F1-Score: 0.8816")

# Initialize the GAT model with best hypertuned parameters
model = AdvancedGAT(
    input_dim=input_dim, 
    hidden_dim=hidden_dim, 
    output_dim=output_dim, 
    heads=heads, 
    dropout=dropout,
    num_layers=num_layers,
    attention_dropout=attention_dropout,
    use_gatv2=True,
    residual=True
)

# Load the saved model state
try:
    model.load_state_dict(torch.load(gat_model_save_path, map_location='cpu'))
    model.eval()
    print(f"\nBest hypertuned GAT model loaded successfully from: {gat_model_save_path}")
except FileNotFoundError:
    print(f"Error: Best hypertuned model file not found at {gat_model_save_path}")
    print("Make sure you've run the hyperparameter tuning script first.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Perform predictions on the entire test set
with torch.no_grad():
    out = model(test_data)
    # Use the same prediction method as training: argmax (no softmax needed)
    _, y_test_pred = out.max(dim=1)
    
    # For analysis, also get the softmax probabilities
    probabilities = F.softmax(out, dim=1)
    probs_class_1 = probabilities[:, 1]

# Debug: Check prediction distribution
unique_preds, counts = torch.unique(y_test_pred, return_counts=True)
print(f"\nPrediction distribution:")
for pred, count in zip(unique_preds, counts):
    print(f"Class {pred}: {count} predictions ({count/len(y_test_pred)*100:.1f}%)")

# Debug: Check probability statistics
print(f"\nProbability statistics for class 1:")
print(f"Min: {probs_class_1.min():.4f}")
print(f"Max: {probs_class_1.max():.4f}")
print(f"Mean: {probs_class_1.mean():.4f}")
print(f"Std: {probs_class_1.std():.4f}")

# Check actual label distribution
unique_labels, label_counts = torch.unique(labels, return_counts=True)
print(f"\nActual label distribution:")
for label, count in zip(unique_labels, label_counts):
    print(f"Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
# Debug: Check raw output statistics
print(f"\nRaw model output statistics:")
print(f"Output shape: {out.shape}")
print(f"Class 0 logits - Min: {out[:, 0].min():.4f}, Max: {out[:, 0].max():.4f}, Mean: {out[:, 0].mean():.4f}")
print(f"Class 1 logits - Min: {out[:, 1].min():.4f}, Max: {out[:, 1].max():.4f}, Mean: {out[:, 1].mean():.4f}")

# Generate a classification report
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT ON TEST SET (Best Hypertuned Model)")
print("Configuration: More Attention Heads (F1=0.8816)")
print(f"{'='*80}")
print(classification_report(labels.cpu(), y_test_pred.cpu(), target_names=["Legitimate (0)", "Illicit (1)"]))

# Calculate F1 scores
f1_macro = f1_score(labels.cpu(), y_test_pred.cpu(), average='macro')
f1_weighted = f1_score(labels.cpu(), y_test_pred.cpu(), average='weighted')
f1_illicit = f1_score(labels.cpu(), y_test_pred.cpu(), pos_label=1, average='binary')
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"F1 Score (Illicit Class): {f1_illicit:.4f}")
print(f"Expected F1 Score (from tuning): 0.8816")

# Display the confusion matrix
conf_matrix = confusion_matrix(labels.cpu(), y_test_pred.cpu())
print(f"\nConfusion Matrix on Test Set:")
print(conf_matrix)
print(f"True Negatives: {conf_matrix[0,0]}, False Positives: {conf_matrix[0,1]}")
print(f"False Negatives: {conf_matrix[1,0]}, True Positives: {conf_matrix[1,1]}")

# Calculate additional metrics
precision_illicit = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) if (conf_matrix[1,1] + conf_matrix[0,1]) > 0 else 0
recall_illicit = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
print(f"\nAdditional Metrics for Illicit Class:")
print(f"Precision: {precision_illicit:.4f}")
print(f"Recall: {recall_illicit:.4f}")

# File paths for saving outputs (updated for best hypertuned model)
graph_visualization_pdf = os.path.join(output_dir, "graph_model_visualization_gat_best_hypertuned.pdf")
performance_plot_path = os.path.join(output_dir, "illicit_f1_by_timestep_best_hypertuned.png")

# Initialize lists to store timestep performance data
timestep_illicit_f1 = []
timestep_numbers = []
timestep_illicit_counts = []
timestep_total_counts = []

# Create a PdfPages object to save all plots
print(f"\nGenerating graph visualizations for best hypertuned model...")
with PdfPages(graph_visualization_pdf) as pdf:
    # Ensure the 'timestep' column exists in the test set
    if 'timestep' not in test_set.columns:
        print("Error: 'timestep' column is missing in the test set.")
    else:
        # Group the test set by timestep
        f1_scores = []
        timesteps = sorted(test_set['timestep'].unique())
        
        print(f"Processing {len(timesteps)} timesteps...")

        for i, timestamp in enumerate(timesteps):
            if i % 5 == 0:  # Print progress every 5 timesteps
                print(f"Processing timestep {i+1}/{len(timesteps)}: {timestamp}")
                
            # Filter the test set for the current timestep
            test_subset = test_set[test_set['timestep'] == timestamp]
            
            if len(test_subset) == 0:
                continue
                
            # Get the actual labels and predictions for this timestep
            timestep_actual_labels = test_subset['class'].values
            timestep_indices = test_subset.index
            
            # Get corresponding predictions (using the mapping from test_set to full predictions)
            # Since we're evaluating on the full test set, we need to map back to the subset
            test_subset_reset = test_subset.reset_index()
            timestep_predictions = []
            
            for _, row in test_subset_reset.iterrows():
                # Find the index in the original test_set
                original_idx = test_set[test_set['transaction_id'] == row['transaction_id']].index[0]
                # Get the corresponding position in our predictions
                pred_idx = list(test_set.index).index(original_idx)
                timestep_predictions.append(y_test_pred[pred_idx].item())
            
            timestep_predictions = np.array(timestep_predictions)
            
            # Calculate illicit F1 score for this timestep
            if len(np.unique(timestep_actual_labels)) > 1 and len(np.unique(timestep_predictions)) > 1:
                illicit_f1_timestep = f1_score(timestep_actual_labels, timestep_predictions, pos_label=1, average='binary', zero_division=0)
            else:
                illicit_f1_timestep = 0.0
            
            # Store timestep performance data
            timestep_illicit_f1.append(illicit_f1_timestep)
            timestep_numbers.append(timestamp)
            timestep_illicit_counts.append(np.sum(timestep_actual_labels == 1))
            timestep_total_counts.append(len(timestep_actual_labels))
            
            transaction_ids = set(test_subset['transaction_id'])
            
            # Filter edgelist for transactions in this timestep
            filtered_edgelist = edgelist_df[
                edgelist_df['txId1'].isin(transaction_ids) & 
                edgelist_df['txId2'].isin(transaction_ids)
            ].copy()

            if len(filtered_edgelist) == 0:
                print(f"No edges found for timestep {timestamp}, skipping graph visualization...")
                continue

            # Create the graph
            G = nx.from_pandas_edgelist(filtered_edgelist, source='txId1', target='txId2', create_using=nx.DiGraph())
            
            if len(G.nodes) == 0:
                continue
                
            # Get node data for this subgraph
            graph_nodes = list(G.nodes)
            graph_nodes_df = pd.DataFrame({'transaction_id': graph_nodes})
            graph_nodes_df = graph_nodes_df.merge(test_subset, on='transaction_id', how='left')

            # Create node feature matrix for this subgraph
            # Apply the same scaling as used for the main test set
            graph_features_raw = graph_nodes_df.drop(columns=['class', 'transaction_id', 'timestep']).fillna(0)
            graph_features_scaled = scaler.transform(graph_features_raw)
            node_features_graph = torch.tensor(graph_features_scaled, dtype=torch.float)

            # Create node mapping for this subgraph
            subgraph_node_mapping = {node_id: idx for idx, node_id in enumerate(graph_nodes)}
            
            # Map edges to new indices
            filtered_edgelist_mapped = filtered_edgelist.copy()
            filtered_edgelist_mapped['txId1_subgraph'] = filtered_edgelist_mapped['txId1'].map(subgraph_node_mapping)
            filtered_edgelist_mapped['txId2_subgraph'] = filtered_edgelist_mapped['txId2'].map(subgraph_node_mapping)
            
            # Remove any edges that couldn't be mapped
            filtered_edgelist_mapped = filtered_edgelist_mapped.dropna(subset=['txId1_subgraph', 'txId2_subgraph'])
            
            if len(filtered_edgelist_mapped) == 0:
                continue
                
            # Convert to PyTorch Geometric format
            subgraph_edge_index = torch.tensor(
                filtered_edgelist_mapped[['txId1_subgraph', 'txId2_subgraph']].astype(int).values.T, 
                dtype=torch.long
            )

            # Add self-loops to subgraph (same as training)
            num_subgraph_nodes = len(node_features_graph)
            subgraph_self_loops = torch.stack([torch.arange(num_subgraph_nodes), torch.arange(num_subgraph_nodes)])
            subgraph_edge_index = torch.cat([subgraph_edge_index, subgraph_self_loops], dim=1)

            # Create the subgraph data object
            data_graph = Data(x=node_features_graph, edge_index=subgraph_edge_index)

            # Perform predictions on this subgraph
            with torch.no_grad():
                out_graph = model(data_graph)
                _, y_graph_pred = out_graph.max(dim=1)  # Same as training

            # Calculate F1 score for this timestep (if we have actual labels)
            actual_labels = graph_nodes_df['class'].dropna()
            if len(actual_labels) > 0:
                # Get predictions for nodes with actual labels
                labeled_indices = graph_nodes_df['class'].dropna().index
                pred_for_labeled = y_graph_pred[labeled_indices]
                f1_timestep = f1_score(actual_labels, pred_for_labeled, average='weighted', zero_division=0)
                f1_scores.append(f1_timestep)

            # Assign colors based on predictions
            node_colors_pred = ['red' if pred == 1 else 'blue' for pred in y_graph_pred]

            # Assign colors based on actual labels (gray for unknown labels)
            node_colors_actual = [
                'red' if pd.notna(actual) and actual == 1 
                else 'blue' if pd.notna(actual) and actual == 0 
                else 'gray'
                for actual in graph_nodes_df['class']
            ]

            # Only create visualization if graph is not too large (for performance)
            if len(G.nodes) <= 500:  # Adjust this threshold as needed
                # Plot the graphs side by side
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))

                # Graph with predicted illicit transactions
                pos = nx.spring_layout(G, k=1, iterations=50)  # Fixed layout for both plots
                nx.draw(
                    G,
                    pos=pos,
                    ax=axes[0],
                    with_labels=False,
                    node_color=node_colors_pred,
                    node_size=50,
                    edge_color='gray',
                    alpha=0.7
                )
                axes[0].set_title(f"Predicted Illicit Transactions (Timestep {timestamp}) - Best Hypertuned Model")

                # Graph with actual illicit transactions
                nx.draw(
                    G,
                    pos=pos,
                    ax=axes[1],
                    with_labels=False,
                    node_color=node_colors_actual,
                    node_size=50,
                    edge_color='gray',
                    alpha=0.7
                )
                axes[1].set_title(f"Actual Illicit Transactions (Timestep {timestamp})")

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='Illicit'),
                    Patch(facecolor='blue', label='Legitimate'),
                    Patch(facecolor='gray', label='Unknown')
                ]
                fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

                plt.tight_layout()
                # Save the current figure to the PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

print(f"\nGraph visualizations saved to: {graph_visualization_pdf}")

# Create the Illicit F1 vs Timestep plot
print(f"\nCreating Illicit F1 vs Timestep plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Illicit F1 Score vs Timestep
ax1.plot(timestep_numbers, timestep_illicit_f1, 'o-', linewidth=2, markersize=6, color='red', alpha=0.7)
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Illicit F1 Score')
ax1.set_title('Best Hypertuned GAT Model: Illicit Class F1 Score by Timestep\n(More Attention Heads - Expected F1: 0.8816)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Add trend line
if len(timestep_numbers) > 1:
    z = np.polyfit(timestep_numbers, timestep_illicit_f1, 1)
    p = np.poly1d(z)
    ax1.plot(timestep_numbers, p(timestep_numbers), "--", alpha=0.8, color='darkred', 
             label=f'Trend (slope: {z[0]:.4f})')
    ax1.legend()

# Plot 2: Illicit Transaction Counts vs Timestep
ax2_twin = ax2.twinx()

# Bar plot for illicit counts
bars = ax2.bar(timestep_numbers, timestep_illicit_counts, alpha=0.6, color='orange', 
               label='Illicit Transactions')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Number of Illicit Transactions', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Line plot for total counts
line = ax2_twin.plot(timestep_numbers, timestep_total_counts, 'b-o', alpha=0.7, 
                     label='Total Transactions')
ax2_twin.set_ylabel('Total Transactions', color='blue')
ax2_twin.tick_params(axis='y', labelcolor='blue')

ax2.set_title('Transaction Counts by Timestep')
ax2.grid(True, alpha=0.3)

# Add combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Illicit F1 vs Timestep plot saved to: {performance_plot_path}")

# Print summary statistics
if f1_scores:
    print(f"\nPer-timestep F1 scores (weighted):")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Std F1: {np.std(f1_scores):.4f}")
    print(f"Min F1: {np.min(f1_scores):.4f}")
    print(f"Max F1: {np.max(f1_scores):.4f}")

if timestep_illicit_f1:
    print(f"\nPer-timestep Illicit F1 scores:")
    print(f"Mean Illicit F1: {np.mean(timestep_illicit_f1):.4f}")
    print(f"Std Illicit F1: {np.std(timestep_illicit_f1):.4f}")
    print(f"Min Illicit F1: {np.min(timestep_illicit_f1):.4f}")
    print(f"Max Illicit F1: {np.max(timestep_illicit_f1):.4f}")
    
    # Calculate correlation between timestep and F1 score
    if len(timestep_numbers) > 1:
        correlation = np.corrcoef(timestep_numbers, timestep_illicit_f1)[0,1]
        print(f"Correlation between timestep and illicit F1: {correlation:.4f}")

print(f"\n{'='*80}")
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("Best Hypertuned Model (More Attention Heads) Results Summary:")
print(f"  - Configuration: hidden_dim=12, heads=8, dropout=0.4, layers=2")
print(f"  - Expected F1 Score: 0.8816")
print(f"  - Actual Illicit F1 Score: {f1_illicit:.4f}")
print(f"  - Macro F1 Score: {f1_macro:.4f}")
print(f"  - Weighted F1 Score: {f1_weighted:.4f}")
print(f"  - Results saved to: {output_dir}")
print(f"{'='*80}")