# elliptic-bitcoin-aml

# Elliptic Bitcoin AML Analysis

This repository contains scripts and models for analyzing and predicting illicit transactions in the Elliptic Bitcoin dataset. The project leverages graph-based features, machine learning models, and advanced graph neural networks to detect illicit activities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

## Overview
The Elliptic Bitcoin AML project focuses on detecting illicit transactions using a combination of:
- Graph-based features (e.g., centrality measures, clustering coefficients).
- Machine learning models like Random Forest.
- Graph Neural Networks (GNNs) such as Graph Attention Networks (GAT).

The dataset includes transaction-level features, graph structures, and class labels (licit, illicit, or unknown).

## Features
- **Feature Engineering**: Extracts graph-based features like betweenness centrality, eigenvector centrality, and ego network properties.
- **Exploratory Data Analysis (EDA)**: Visualizes data distributions, correlations, and graph structures.
- **Model Training**:
  - Random Forest with hyperparameter tuning.
  - Graph Attention Networks (GAT) with advanced configurations.
- **Visualization**: Generates side-by-side graphs comparing predicted and actual illicit transactions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/elliptic-bitcoin-aml.git
   cd elliptic-bitcoin-aml

Then please execute the files in the following order:
1.- requirements.txt 
2.- download_data.py - simple script to download the Elliptic dataset from Kaggle. Make sure to have Kaggle API installed and configured
3.- exploratory_data_analysis.py - This script performs exploratory data analysis (EDA) on the Elliptic Bitcoin dataset. It includes loading the dataset, checking for missing values, visualizing distributions, and analyzing correlations.
4.- feature_engineering.py This script creates features for the prediction of illicit activity in the elliptic dataset. The features are based on the graph structure of the transactions 
5.- anomaly_detection.py This script will conduct the analysis of the association between the features created in the feature engineering script and available in the dataset, and the target variable.
6.- model_selection.py This script will adjust the proposed models to the data and evaluate their performance.
7.- gat_model.py. This script trains and tunes a GAT model to the training data. 
8.- final_model_gat.py Assessment of the final GAT model performance on the test set. 
9.- final_model_rf.py Assessment of the final random forest model performance on the test set. Visualization of the features relationship with the target. Visualization of the graphs comparing the model prediction with the actual labels. 
10. data_drift.py Exploration of data drift around the regulatory change at timestep 43. 


#Project Structure 
elliptic-bitcoin-aml/
├── data/
│   └── elliptic_bitcoin_dataset/  # Dataset files
├── outputs/                       # Generated outputs (e.g., models, plots)
├── scripts/                       # Python scripts for analysis and modeling
│   ├── anomaly_detection.py
│   ├── data_drift.py
│   ├── exploratory_data_analysis.py
│   ├── feature_engineering.py
│   ├── final_model_rf.py
│   ├── final_model_gat.py
│   ├── model_selection.py
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
