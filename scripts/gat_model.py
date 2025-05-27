import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm, GATv2Conv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(1001)
np.random.seed(1001)
random.seed(1001)

# Advanced GAT Model (same as your original)
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
        
        conv_layer = GATv2Conv if use_gatv2 else GATConv
        
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

# Enhanced training function with flexible scheduler
def train_and_evaluate_advanced(model, data, criterion, optimizer, scheduler, scheduler_type,
                               train_mask, val_mask, max_epochs=500, patience=50):
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation check every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                _, val_pred = val_out.max(dim=1)
                val_labels = data.y[val_mask].cpu().numpy()
                val_preds = val_pred[val_mask].cpu().numpy()
                
                # Focus on illicit F1-score for early stopping
                val_f1_illicit = f1_score(val_labels, val_preds, labels=[1], average=None)
                val_f1_illicit = val_f1_illicit[0] if len(val_f1_illicit) > 0 else 0
            
            # Update scheduler based on type
            if scheduler_type == 'cosine':
                scheduler.step()
            elif scheduler_type == 'plateau':
                scheduler.step(val_f1_illicit)
            
            # Early stopping based on F1-score
            if val_f1_illicit > best_val_f1:
                best_val_f1 = val_f1_illicit
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience // 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# Load the data (same as your original code)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "outputs")
data_dir = os.path.join(base_dir, "data", "elliptic_bitcoin_dataset")

os.makedirs(output_dir, exist_ok=True)

train_set_clean_path = os.path.join(output_dir, "train_set_clean.csv")
edgelist_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
train_set_clean = pd.read_csv(train_set_clean_path)
edgelist_df = pd.read_csv(edgelist_file)

# Load original data first (before oversampling for edge processing)
original_train_set = pd.read_csv(train_set_clean_path)

# Data preprocessing with original data for edges
transaction_to_timestep = original_train_set.set_index('transaction_id')['timestep'].to_dict()
edgelist_df['timestep1'] = edgelist_df['txId1'].map(transaction_to_timestep)
edgelist_df['timestep2'] = edgelist_df['txId2'].map(transaction_to_timestep)

valid_transactions = set(original_train_set['transaction_id'].unique())
edgelist_df = edgelist_df[
    edgelist_df['txId1'].isin(valid_transactions) & 
    edgelist_df['txId2'].isin(valid_transactions)
]

# Create node mapping based on original data
original_node_id_mapping = {node_id: idx for idx, node_id in enumerate(original_train_set['transaction_id'])}
edgelist_df['txId1_mapped'] = edgelist_df['txId1'].map(original_node_id_mapping)
edgelist_df['txId2_mapped'] = edgelist_df['txId2'].map(original_node_id_mapping)
edgelist_df = edgelist_df.dropna(subset=['txId1_mapped', 'txId2_mapped'])
edgelist_df[['txId1_mapped', 'txId2_mapped']] = edgelist_df[['txId1_mapped', 'txId2_mapped']].astype(int)

feature_columns = [col for col in train_set_clean.columns if col not in ['class', 'transaction_id', 'timestep']]

# Apply 70% licit, 30% illicit oversampling
print("🔄 Applying oversampling to achieve 70% licit, 30% illicit distribution...")
print(f"Original class distribution:")
original_counts = train_set_clean['class'].value_counts().sort_index()
print(f"  Licit (0): {original_counts[0]} ({original_counts[0]/len(train_set_clean)*100:.1f}%)")
print(f"  Illicit (1): {original_counts[1]} ({original_counts[1]/len(train_set_clean)*100:.1f}%)")

# Calculate target counts for 70% licit, 30% illicit
total_target = len(train_set_clean)
target_licit = int(total_target * 0.7)
target_illicit = int(total_target * 0.3)

# Use the existing licit count and oversample illicit to match desired ratio
current_licit = original_counts[0]
if target_illicit > original_counts[1]:
    # We need to oversample illicit class
    sampling_strategy = {0: current_licit, 1: int(current_licit * 0.3 / 0.7)}
else:
    # If we already have enough illicit, keep original counts
    sampling_strategy = 'auto'

# Apply oversampling
X_features = train_set_clean[feature_columns + ['transaction_id', 'timestep']]
y_labels = train_set_clean['class']

ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=1001)
X_resampled, y_resampled = ros.fit_resample(X_features, y_labels)

# Create oversampled dataset
train_set_oversampled = pd.DataFrame(X_resampled, columns=feature_columns + ['transaction_id', 'timestep'])
train_set_oversampled['class'] = y_resampled

print(f"After oversampling:")
new_counts = train_set_oversampled['class'].value_counts().sort_index()
print(f"  Licit (0): {new_counts[0]} ({new_counts[0]/len(train_set_oversampled)*100:.1f}%)")
print(f"  Illicit (1): {new_counts[1]} ({new_counts[1]/len(train_set_oversampled)*100:.1f}%)")
print(f"  Total samples: {len(train_set_oversampled)}")

# Update the dataset reference
train_set_clean = train_set_oversampled.copy()

# 10 Carefully Selected Hyperparameter Configurations
# Designed to avoid overfitting while exploring different aspects
hyperparameter_configs = [
    {
        "name": "Baseline Conservative",
        "hidden_dim": 16, "heads": 4, "lr": 0.01, "dropout": 0.5, "num_layers": 2,
        "weight_decay": 1e-3, "scheduler": "cosine", "optimizer": "adamw"
    },
    {
        "name": "Higher Dropout",
        "hidden_dim": 20, "heads": 6, "lr": 0.008, "dropout": 0.6, "num_layers": 2,
        "weight_decay": 5e-4, "scheduler": "plateau", "optimizer": "adamw"
    },
    {
        "name": "More Attention Heads",
        "hidden_dim": 12, "heads": 8, "lr": 0.012, "dropout": 0.4, "num_layers": 2,
        "weight_decay": 8e-4, "scheduler": "cosine", "optimizer": "adamw"
    },
    {
        "name": "Moderate Complexity",
        "hidden_dim": 24, "heads": 6, "lr": 0.009, "dropout": 0.45, "num_layers": 3,
        "weight_decay": 1e-3, "scheduler": "plateau", "optimizer": "adam"
    },
    {
        "name": "Low Learning Rate",
        "hidden_dim": 18, "heads": 8, "lr": 0.005, "dropout": 0.35, "num_layers": 2,
        "weight_decay": 1.5e-3, "scheduler": "cosine", "optimizer": "adamw"
    },
    {
        "name": "High Regularization",
        "hidden_dim": 20, "heads": 6, "lr": 0.01, "dropout": 0.55, "num_layers": 2,
        "weight_decay": 2e-3, "scheduler": "plateau", "optimizer": "adamw"
    },
    {
        "name": "Balanced Medium",
        "hidden_dim": 22, "heads": 7, "lr": 0.007, "dropout": 0.4, "num_layers": 3,
        "weight_decay": 7e-4, "scheduler": "cosine", "optimizer": "adam"
    },
    {
        "name": "Small but Deep",
        "hidden_dim": 14, "heads": 6, "lr": 0.011, "dropout": 0.5, "num_layers": 3,
        "weight_decay": 1.2e-3, "scheduler": "plateau", "optimizer": "adamw"
    },
    {
        "name": "Wide but Shallow",
        "hidden_dim": 28, "heads": 8, "lr": 0.006, "dropout": 0.5, "num_layers": 2,
        "weight_decay": 1.5e-3, "scheduler": "cosine", "optimizer": "adamw"
    },
    {
        "name": "Conservative Plus",
        "hidden_dim": 16, "heads": 5, "lr": 0.008, "dropout": 0.45, "num_layers": 2,
        "weight_decay": 9e-4, "scheduler": "plateau", "optimizer": "adam"
    }
]

best_f1 = 0.0
best_params = None
best_model = None
all_results = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

print("🔍 Starting Hyperparameter Tuning with 10 Configurations")
print("📊 Using 70% Licit, 30% Illicit Oversampling (No Weighted Loss)")
print("=" * 80)

for i, params in enumerate(hyperparameter_configs):
    print(f"\n📊 Configuration {i+1}/10: {params['name']}")
    print(f"Parameters: hidden_dim={params['hidden_dim']}, heads={params['heads']}, "
          f"lr={params['lr']}, dropout={params['dropout']}, layers={params['num_layers']}")
    print(f"Regularization: weight_decay={params['weight_decay']}, scheduler={params['scheduler']}")
    print("-" * 60)
    
    # Prepare oversampled data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(train_set_clean[feature_columns])
    node_features = torch.tensor(normalized_features, dtype=torch.float)
    labels = torch.tensor(train_set_clean['class'].values, dtype=torch.long)
    
    # Create new node mapping for oversampled data
    oversampled_node_mapping = {node_id: idx for idx, node_id in enumerate(train_set_clean['transaction_id'])}
    
    # Map edges to oversampled node indices
    edge_list = []
    for _, row in edgelist_df.iterrows():
        src_id = row['txId1']
        dst_id = row['txId2']
        
        # Find all occurrences of these transaction IDs in oversampled data
        src_indices = train_set_clean[train_set_clean['transaction_id'] == src_id].index.tolist()
        dst_indices = train_set_clean[train_set_clean['transaction_id'] == dst_id].index.tolist()
        
        # Create edges between all combinations
        for src_idx in src_indices:
            for dst_idx in dst_indices:
                edge_list.append([src_idx, dst_idx])
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    else:
        # Fallback: create basic edges if no valid edges found
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Add self-loops
    num_nodes = len(node_features)
    self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    
    # Use standard CrossEntropyLoss (no weighting since we oversampled)
    criterion = torch.nn.CrossEntropyLoss()
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(node_features, labels)):
        print(f"  Fold {fold + 1}/5", end=" ")
        
        # Split train into train/val
        train_size = int(0.8 * len(train_idx))
        train_fold_idx = train_idx[:train_size]
        val_fold_idx = train_idx[train_size:]
        
        # Create masks
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)
        
        train_mask[train_fold_idx] = True
        val_mask[val_fold_idx] = True
        test_mask[test_idx] = True
        
        # Initialize model
        model = AdvancedGAT(
            input_dim=node_features.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=2,
            heads=params['heads'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            attention_dropout=params['dropout'] * 0.5,
            use_gatv2=True,
            residual=True
        )
        
        # Choose optimizer
        if params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay'],
                amsgrad=True
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
        
        # Choose scheduler
        if params['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=params['lr']*0.01)
        else:  # plateau
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, 
                                        min_lr=params['lr']*0.001)
        
        # Train model
        model = train_and_evaluate_advanced(
            model, data, criterion, optimizer, scheduler, params['scheduler'],
            train_mask, val_mask, max_epochs=250, patience=35
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data)
            _, pred = out.max(dim=1)
        
        # Calculate metrics
        test_labels = labels[test_mask].cpu().numpy()
        test_preds = pred[test_mask].cpu().numpy()
        
        f1_illicit = f1_score(test_labels, test_preds, labels=[1], average=None)
        f1_illicit = f1_illicit[0] if len(f1_illicit) > 0 else 0
        
        fold_results.append(f1_illicit)
        print(f"F1: {f1_illicit:.4f}")
    
    # Calculate statistics
    avg_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    
    # Store results
    result = {
        'config_name': params['name'],
        'params': params.copy(),
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'fold_results': fold_results.copy()
    }
    all_results.append(result)
    
    print(f"  💯 Average F1 (Illicit): {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Check if this is the best so far
    if avg_f1 > best_f1:
        improvement = avg_f1 - best_f1
        print(f"  🚀 NEW BEST! Improved by +{improvement:.4f} (from {best_f1:.4f} to {avg_f1:.4f})")
        best_f1 = avg_f1
        best_params = params.copy()
        best_model = model
    else:
        difference = best_f1 - avg_f1
        print(f"  📉 Below best: -{difference:.4f}")

# Final Results Summary
print(f"\n{'='*80}")
print(" 🏆 HYPERPARAMETER TUNING RESULTS")
print(f"{'='*80}")

# Sort results by F1 score
all_results.sort(key=lambda x: x['avg_f1'], reverse=True)

print("\n📊 TOP 5 CONFIGURATIONS:")
print("-" * 60)
for i, result in enumerate(all_results[:5]):
    print(f"{i+1}. {result['config_name']}: {result['avg_f1']:.4f} ± {result['std_f1']:.4f}")
    key_params = f"h={result['params']['hidden_dim']}, heads={result['params']['heads']}, lr={result['params']['lr']:.3f}, dropout={result['params']['dropout']}"
    print(f"   {key_params}")

print(f"\n🥇 BEST CONFIGURATION: {best_params['name']}")
print(f"📈 Best F1-Score: {best_f1:.4f}")
print(f"🔧 Best Parameters:")
for key, value in best_params.items():
    if key != 'name':
        print(f"   {key}: {value}")

# Save best model
if best_model is not None:
    model_save_path = os.path.join(output_dir, "best_hypertuned_gat_model.pth")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"💾 Best model saved to {model_save_path}")

# Save detailed results
results_df = pd.DataFrame([{
    'rank': i+1,
    'config_name': r['config_name'],
    'avg_f1': r['avg_f1'],
    'std_f1': r['std_f1'],
    **r['params']
} for i, r in enumerate(all_results)])

results_path = os.path.join(output_dir, "hyperparameter_tuning_results.csv")
results_df.to_csv(results_path, index=False)
print(f"📄 Detailed results saved to {results_path}")

print(f"\n{'='*80}")
print("🎯 Hyperparameter tuning completed successfully!")
print(f"{'='*80}")