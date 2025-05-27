import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Key network features for focused analysis
key_network_features = ['betweenness_centrality', 'avg_shortest_path_length', 'clustering_coefficient_x', 
                       'eigenvector_centrality', 'density', 'avg_degree']

# Top features from your analysis
top_features = ['feature_53', 'feature_55', 'feature_14', 'feature_138', 'feature_49', 'feature_41', 
               'feature_5', 'feature_132', 'feature_47', 'feature_90', 'feature_29', 'feature_18']

# Define base paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "elliptic_bitcoin_dataset")
output_dir = os.path.join(base_dir, "outputs")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File paths
train_set_clean_path = os.path.join(output_dir, "train_set_clean.csv")
test_set_path = os.path.join(output_dir, "test_set.csv")

def create_side_by_side_histograms(before_data, after_data, features_to_plot, output_dir, split_timestep=43):
    """Create side-by-side transparent histograms for feature comparison"""
    
    print(f"Creating side-by-side histograms for {len(features_to_plot)} features...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create PDF for all histograms
    pdf_path = os.path.join(output_dir, f"side_by_side_histograms_timestep_{split_timestep}.pdf")
    
    # Also create individual PNG files
    png_dir = os.path.join(output_dir, "histogram_pngs")
    os.makedirs(png_dir, exist_ok=True)
    
    with PdfPages(pdf_path) as pdf:
        
        # Create overview plot with multiple features
        n_features = min(9, len(features_to_plot))  # Max 9 features for 3x3 grid
        if n_features > 0:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Feature Distributions: Before vs After Timestep {split_timestep} (Overview)', 
                        fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for i, feature in enumerate(features_to_plot[:n_features]):
                if feature not in before_data.columns or feature not in after_data.columns:
                    print(f"Skipping {feature} - not found in data")
                    continue
                
                # Get data and remove NaN values
                before_vals = before_data[feature].dropna()
                after_vals = after_data[feature].dropna()
                
                if len(before_vals) == 0 or len(after_vals) == 0:
                    print(f"Skipping {feature} - no valid data")
                    continue
                
                # Create histogram with transparency
                axes[i].hist(before_vals, bins=50, alpha=0.6, label=f'Before {split_timestep}', 
                           color='blue', density=True, edgecolor='navy', linewidth=0.5)
                axes[i].hist(after_vals, bins=50, alpha=0.6, label=f'After {split_timestep}', 
                           color='red', density=True, edgecolor='darkred', linewidth=0.5)
                
                # Formatting
                axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Density')
                axes[i].legend(loc='upper right', fontsize=9)
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics text
                before_mean = before_vals.mean()
                after_mean = after_vals.mean()
                mean_change = ((after_mean - before_mean) / before_mean * 100) if before_mean != 0 else 0
                
                stats_text = f'Δμ: {mean_change:+.1f}%'
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                           fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8))
            
            # Hide empty subplots
            for i in range(n_features, 9):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(png_dir, 'overview_histograms.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create detailed individual plots for each feature
        for feature in features_to_plot:
            if feature not in before_data.columns or feature not in after_data.columns:
                print(f"Skipping {feature} - not found in data")
                continue
            
            try:
                # Get data and remove NaN values
                before_vals = before_data[feature].dropna()
                after_vals = after_data[feature].dropna()
                
                if len(before_vals) == 0 or len(after_vals) == 0:
                    print(f"Skipping {feature} - no valid data")
                    continue
                
                print(f"Processing {feature}: Before={len(before_vals)}, After={len(after_vals)}")
                
                # Create detailed plot for this feature
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Detailed Analysis: {feature}', fontsize=16, fontweight='bold')
                
                # Main histogram with transparency
                axes[0, 0].hist(before_vals, bins=60, alpha=0.7, label=f'Before {split_timestep}', 
                              color='skyblue', density=True, edgecolor='navy', linewidth=0.8)
                axes[0, 0].hist(after_vals, bins=60, alpha=0.7, label=f'After {split_timestep}', 
                              color='salmon', density=True, edgecolor='darkred', linewidth=0.8)
                
                axes[0, 0].set_title('Overlapping Histograms (Density)', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Value')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].legend(loc='best')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Box plot comparison
                box_data = [before_vals, after_vals]
                bp = axes[0, 1].boxplot(box_data, labels=[f'Before {split_timestep}', f'After {split_timestep}'], 
                                       patch_artist=True, widths=0.6)
                bp['boxes'][0].set_facecolor('skyblue')
                bp['boxes'][1].set_facecolor('salmon')
                axes[0, 1].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Cumulative distribution
                axes[1, 0].hist(before_vals, bins=100, alpha=0.8, label=f'Before {split_timestep}', 
                              density=True, cumulative=True, histtype='step', linewidth=3, color='blue')
                axes[1, 0].hist(after_vals, bins=100, alpha=0.8, label=f'After {split_timestep}', 
                              density=True, cumulative=True, histtype='step', linewidth=3, color='red')
                axes[1, 0].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].set_ylabel('Cumulative Probability')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Statistics summary
                before_stats = before_vals.describe()
                after_stats = after_vals.describe()
                
                # Calculate additional metrics
                mean_change = ((after_stats['mean'] - before_stats['mean']) / before_stats['mean'] * 100) if before_stats['mean'] != 0 else 0
                std_change = ((after_stats['std'] - before_stats['std']) / before_stats['std'] * 100) if before_stats['std'] != 0 else 0
                
                stats_text = f"""
BEFORE TIMESTEP {split_timestep}:
  Count: {len(before_vals):,}
  Mean:  {before_stats['mean']:.6f}
  Std:   {before_stats['std']:.6f}
  Min:   {before_stats['min']:.6f}
  25%:   {before_stats['25%']:.6f}
  50%:   {before_stats['50%']:.6f}
  75%:   {before_stats['75%']:.6f}
  Max:   {before_stats['max']:.6f}

AFTER TIMESTEP {split_timestep}:
  Count: {len(after_vals):,}
  Mean:  {after_stats['mean']:.6f}
  Std:   {after_stats['std']:.6f}
  Min:   {after_stats['min']:.6f}
  25%:   {after_stats['25%']:.6f}
  50%:   {after_stats['50%']:.6f}
  75%:   {after_stats['75%']:.6f}
  Max:   {after_stats['max']:.6f}

CHANGES:
  Mean Change:  {mean_change:+.2f}%
  Std Change:   {std_change:+.2f}%
                """
                
                axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                               fontsize=9, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                
                # Save individual PNG
                png_filename = f'{feature}_histogram_comparison.png'
                plt.savefig(os.path.join(png_dir, png_filename), dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error processing {feature}: {e}")
                continue
    
    print(f"\nHistogram visualizations saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNGs: {png_dir}/")

def load_and_split_data():
    """Load data and split by timestep 43"""
    
    print("Loading datasets...")
    
    # Load train and test sets
    train_data = pd.read_csv(train_set_clean_path)
    test_data = pd.read_csv(test_set_path)
    
    print(f"Train set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")
    
    # Check if timestep column exists
    if 'timestep' not in test_data.columns:
        print("Error: 'timestep' column not found in test data")
        return None, None, None
    
    # Combine train and test data for complete analysis
    if 'timestep' in train_data.columns:
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
    else:
        combined_data = test_data.copy()
        print("No timestep in train data, using only test data for analysis")
    
    # Check if timestep 43 exists in the data
    available_timesteps = sorted(combined_data['timestep'].unique())
    print(f"Available timesteps: {min(available_timesteps)} to {max(available_timesteps)}")
    
    if 43 not in available_timesteps:
        print(f"Warning: Timestep 43 not found in data. Using median timestep {np.median(available_timesteps):.0f} instead.")
        split_timestep = int(np.median(available_timesteps))
    else:
        split_timestep = 43
    
    # Split data into before and after the split timestep
    before_split = combined_data[combined_data['timestep'] < split_timestep]
    after_split = combined_data[combined_data['timestep'] >= split_timestep]
    
    print(f"Data before timestep {split_timestep}: {len(before_split)} transactions")
    print(f"Data after timestep {split_timestep} (inclusive): {len(after_split)} transactions")
    
    if len(before_split) == 0 or len(after_split) == 0:
        print("Error: One of the periods has no data. Cannot perform comparison.")
        return None, None, None
    
    return before_split, after_split, split_timestep

def main():
    """Main function to create side-by-side histograms"""
    
    print("="*60)
    print("SIDE-BY-SIDE HISTOGRAM ANALYSIS: BEFORE vs AFTER TIMESTEP 43")
    print("="*60)
    
    # Load and split data
    before_data, after_data, split_timestep = load_and_split_data()
    
    if before_data is None:
        print("Could not load or split data. Exiting.")
        return
    
    # Get available features
    available_network_features = [f for f in key_network_features 
                                 if f in before_data.columns and f in after_data.columns]
    available_top_features = [f for f in top_features 
                             if f in before_data.columns and f in after_data.columns]
    
    print(f"Available network features: {len(available_network_features)}")
    print(f"Available top features: {len(available_top_features)}")
    
    # Combine features for analysis (prioritize network features)
    features_to_analyze = available_network_features + available_top_features[:6]  # Limit to manageable number
    
    print(f"Features to analyze: {features_to_analyze}")
    
    # Create histograms
    create_side_by_side_histograms(before_data, after_data, features_to_analyze, output_dir, split_timestep)
    
    print(f"\n🎯 HISTOGRAM ANALYSIS COMPLETE!")
    print(f"   • Analyzed {len(features_to_analyze)} features")
    print(f"   • Before period: {len(before_data):,} transactions")
    print(f"   • After period: {len(after_data):,} transactions")
    print(f"   • Files saved in: {output_dir}")

if __name__ == "__main__":
    main()